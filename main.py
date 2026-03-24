import streamlit as st
import os
import tempfile
import shutil
import pandas as pd
import geopandas as gpd
import numpy as np
from PIL import Image
import json
import zipfile
from io import BytesIO

# Librerías para mapas interactivos
import folium
from streamlit_folium import st_folium

# MapReader y PyTorch
import mapreader
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from mapreader import MapImages
from mapreader.classify.load_annotations import AnnotationsLoader
from mapreader.classify.classifier import ClassifierContainer

# Configuración de la página
st.set_page_config(layout="wide", page_title="Análisis de Tierras en Patagonia")
st.title("🗺️ Análisis de Asignación de Tierras en la Patagonia")
st.markdown("Sube mapas históricos, anota parcelas, entrena un modelo y georreferencia los resultados con Allmaps.")

# -------------------------------------------------------------------
# Inicialización de variables de sesión
# -------------------------------------------------------------------
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.archivos_guardados = []

if "map_images" not in st.session_state:
    st.session_state.map_images = None
if "df_patches" not in st.session_state:
    st.session_state.df_patches = None
if "annotations_file" not in st.session_state:
    st.session_state.annotations_file = None
if "classifier" not in st.session_state:
    st.session_state.classifier = None
if "labels_map" not in st.session_state:
    st.session_state.labels_map = None
if "predictions_df" not in st.session_state:
    st.session_state.predictions_df = None
if "geojson_output" not in st.session_state:
    st.session_state.geojson_output = None
if "map_names" not in st.session_state:
    st.session_state.map_names = []

# -------------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------------
def pixel_to_geo(pixel_x, pixel_y, gcps):
    if len(gcps) < 3:
        return (None, None)
    n = len(gcps)
    A = np.zeros((2*n, 6))
    B_lon = np.zeros(2*n)
    B_lat = np.zeros(2*n)
    for i, gcp in enumerate(gcps):
        px, py = gcp['pixel']
        lon, lat = gcp['geo']
        A[2*i, 0] = px
        A[2*i, 1] = py
        A[2*i, 2] = 1
        B_lon[2*i] = lon
        A[2*i+1, 3] = px
        A[2*i+1, 4] = py
        A[2*i+1, 5] = 1
        B_lat[2*i+1] = lat
    try:
        coeffs_lon, _, _, _ = np.linalg.lstsq(A, B_lon, rcond=1e-6)
        coeffs_lat, _, _, _ = np.linalg.lstsq(A, B_lat, rcond=1e-6)
    except np.linalg.LinAlgError:
        return (None, None)
    lon = coeffs_lon[0]*pixel_x + coeffs_lon[1]*pixel_y + coeffs_lon[2]
    lat = coeffs_lat[3]*pixel_x + coeffs_lat[4]*pixel_y + coeffs_lat[5]
    if lon < -180 or lon > 180 or lat < -90 or lat > 90:
        return (None, None)
    return (lon, lat)

def parse_allmaps_manifest(manifest_json):
    gcps = []
    try:
        items = manifest_json.get("items", [])
        if items:
            annotation = items[0]
        else:
            annotation = manifest_json
        body = annotation.get("body", {})
        if body.get("type") == "FeatureCollection":
            features = body.get("features", [])
            for feature in features:
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})
                resource_coords = props.get("resourceCoords", [])
                geo_coords = geom.get("coordinates", [])
                if resource_coords and len(resource_coords) == 2 and geo_coords and len(geo_coords) == 2:
                    gcps.append({
                        "pixel": (resource_coords[0], resource_coords[1]),
                        "geo": (geo_coords[0], geo_coords[1])
                    })
    except Exception as e:
        st.warning(f"Error al parsear el manifiesto: {e}")
    return gcps

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.header("1. Carga de mapas")
    uploaded_files = st.file_uploader(
        "Selecciona uno o más mapas (imágenes)",
        type=["jpg", "jpeg", "png", "tiff"],
        accept_multiple_files=True
    )

    patch_size = st.slider("Tamaño del parche (px) - Recomendado 200-300 para más muestras", 100, 1000, 200, key="patch_slider")

    if uploaded_files and st.button("📥 Procesar mapas (generar parches)"):
        nombres_mapas = []
        for uploaded_file in uploaded_files:
            path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
            if path not in st.session_state.archivos_guardados:
                st.session_state.archivos_guardados.append(path)
            nombres_mapas.append(uploaded_file.name)
        st.session_state.map_names = nombres_mapas

        with st.spinner("Cargando imágenes..."):
            st.session_state.map_images = MapImages(st.session_state.temp_dir)

        with st.spinner(f"Parcheando mapas en fragmentos de {patch_size}px..."):
            st.session_state.map_images.patchify_all(patch_size=patch_size)

            patches_dict = st.session_state.map_images.images['patch']

            patch_data = []
            for patch_id, patch in patches_dict.items():
                pixel_bounds = patch['pixel_bounds']
                if isinstance(pixel_bounds, tuple):
                    min_x = pixel_bounds[0]
                    min_y = pixel_bounds[1]
                else:
                    min_x = pixel_bounds.get('min_x', 0)
                    min_y = pixel_bounds.get('min_y', 0)

                patch_data.append({
                    "patch_id": patch_id,
                    "parent_id": patch['parent_id'],
                    "min_x": min_x,
                    "min_y": min_y,
                    "image_path": patch['image_path']
                })
            st.session_state.df_patches = pd.DataFrame(patch_data)

        st.success(f"✅ Se generaron {len(st.session_state.df_patches)} parches.")
        st.write("Columnas del DataFrame:", st.session_state.df_patches.columns.tolist())

    st.header("2. Anotaciones")
    if st.session_state.df_patches is not None:
        opcion_anot = st.radio("Tipo de anotación", ["Usar demostración", "Subir mi propio CSV"])

        if opcion_anot == "Usar demostración":
            if st.button("✏️ Crear anotaciones de demostración"):
                num_samples = min(20, len(st.session_state.df_patches))
                sample_df = st.session_state.df_patches.head(num_samples).copy()
                labels = ["concesion", "limite", "fondo", "ferrocarril"]
                sample_df['label'] = [labels[i % 4] for i in range(len(sample_df))]
                csv_path = os.path.join(st.session_state.temp_dir, "anotaciones_demo.csv")
                sample_df[['image_path', 'label']].to_csv(csv_path, index=False)
                st.session_state.annotations_file = csv_path
                st.success(f"Archivo de anotaciones de ejemplo creado ({len(sample_df)} parches).")

                df_view = pd.read_csv(csv_path)
                st.subheader("Vista previa de las anotaciones")
                st.dataframe(df_view)

        else:
            uploaded_csv = st.file_uploader("Sube tu archivo CSV (columnas: image_path, label)", type=["csv"])
            if uploaded_csv and st.button("Cargar CSV"):
                csv_path = os.path.join(st.session_state.temp_dir, "mis_anotaciones.csv")
                with open(csv_path, "wb") as f:
                    f.write(uploaded_csv.getvalue())
                st.session_state.annotations_file = csv_path
                st.success("CSV cargado correctamente.")

                df_view = pd.read_csv(csv_path)
                st.subheader("Vista previa de las anotaciones")
                st.dataframe(df_view)
    else:
        st.info("Primero carga y procesa mapas.")

    st.header("3. Entrenar modelo")
    if st.session_state.df_patches is not None and st.session_state.annotations_file is not None:
        if st.checkbox("Mostrar vista previa del archivo de anotaciones"):
            try:
                df_annot_preview = pd.read_csv(st.session_state.annotations_file)
                st.dataframe(df_annot_preview.head(10))
                st.write("Columnas:", df_annot_preview.columns.tolist())
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")

        epochs = st.number_input("Número de épocas", min_value=1, max_value=50, value=5, step=1, key="epochs")

        if st.button("🚀 Entrenar clasificador"):
            try:
                df_annot = pd.read_csv(st.session_state.annotations_file)
            except Exception as e:
                st.error(f"Error al leer el archivo CSV: {e}")
                st.stop()

            if 'image_path' not in df_annot.columns:
                st.error("El archivo CSV debe contener una columna 'image_path' con las rutas de los parches.")
                st.write("Columnas encontradas:", df_annot.columns.tolist())
                st.stop()
            if 'label' not in df_annot.columns:
                st.error("El archivo CSV debe contener una columna 'label' con las etiquetas.")
                st.stop()

            min_samples_per_class = df_annot['label'].value_counts().min()
            if min_samples_per_class < 2:
                st.warning("Hay clases con muy pocas muestras. El entrenamiento puede fallar. Intenta aumentar el número de parches o reducir el tamaño del parche.")

            with st.spinner("Cargando anotaciones y entrenando (puede tomar varios minutos)..."):
                loader = AnnotationsLoader()
                loader.load(
                    annotations=df_annot,
                    patch_paths_col="image_path",
                    label_col="label"
                )

                total_samples = len(df_annot)
                min_samples_per_class = df_annot['label'].value_counts().min()
                # --- AJUSTE: usar solo train/val si hay pocos datos o clases pequeñas ---
                if total_samples < 30 or min_samples_per_class < 5:
                    st.info("Pocos datos o clases con pocas muestras: se usará 80% entrenamiento, 20% validación (sin test).")
                    loader.create_datasets(frac_train=0.8, frac_val=0.2, frac_test=0.0)
                else:
                    loader.create_datasets(frac_train=0.7, frac_val=0.15, frac_test=0.15)

                dataloaders = loader.create_dataloaders(batch_size=16)

                st.session_state.classifier = ClassifierContainer(
                    model="resnet18",
                    dataloaders=dataloaders,
                    labels_map=loader.labels_map
                )
                st.session_state.classifier.add_loss_fn("cross_entropy")
                st.session_state.classifier.initialize_optimizer("adam")
                st.session_state.classifier.train(num_epochs=epochs)

                st.session_state.labels_map = loader.labels_map
            st.success("✅ Modelo entrenado.")
    else:
        st.info("Necesitas anotaciones primero (paso 2).")

    st.header("4. Predecir en todos los parches")
    if st.session_state.classifier is not None:
        if st.button("🔍 Ejecutar predicción"):
            with st.spinner("Clasificando todos los parches..."):
                image_paths = st.session_state.df_patches['image_path'].tolist()

                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                class PatchDataset(Dataset):
                    def __init__(self, paths, transform):
                        self.paths = paths
                        self.transform = transform
                    def __len__(self):
                        return len(self.paths)
                    def __getitem__(self, idx):
                        img = Image.open(self.paths[idx]).convert('RGB')
                        img = self.transform(img)
                        return img, self.paths[idx]

                dataset = PatchDataset(image_paths, transform)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

                model = st.session_state.classifier.model
                model.eval()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)

                all_probs = []
                all_paths = []
                with torch.no_grad():
                    for batch, paths in dataloader:
                        batch = batch.to(device)
                        outputs = model(batch)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        all_probs.append(probs.cpu())
                        all_paths.extend(paths)

                probs = torch.cat(all_probs, dim=0)
                pred_labels_idx = probs.argmax(dim=1).numpy()
                pred_probs = probs.max(dim=1)[0].numpy()

                idx_to_label = {v: k for k, v in st.session_state.labels_map.items()}
                pred_labels = [idx_to_label.get(idx, "desconocido") for idx in pred_labels_idx]

                predictions = pd.DataFrame({
                    'image_path': all_paths,
                    'predicted_label': pred_labels,
                    'probability': pred_probs
                })
                st.session_state.predictions_df = predictions
            st.success(f"✅ Predicción completada. {len(predictions)} parches clasificados.")
    else:
        st.info("Entrena un modelo primero (paso 3).")

    st.header("5. Integración con Allmaps")
    if st.session_state.predictions_df is not None:
        if st.session_state.map_names:
            mapa_seleccionado = st.selectbox("Selecciona el mapa a georreferenciar", st.session_state.map_names)
        else:
            mapa_seleccionado = None
            st.warning("No hay mapas registrados. Sube mapas en el paso 1.")

        manifest_file = st.file_uploader("Sube el manifiesto JSON de Allmaps", type=["json"])
        if manifest_file and mapa_seleccionado and st.button("🌍 Generar GeoJSON"):
            manifest = json.load(manifest_file)
            gcps = parse_allmaps_manifest(manifest)

            if len(gcps) < 3:
                st.error("El manifiesto no contiene suficientes puntos de control (necesita al menos 3).")
            else:
                needed_cols = ['image_path', 'parent_id', 'min_x', 'min_y']
                missing = [col for col in needed_cols if col not in st.session_state.df_patches.columns]
                if not missing:
                    merged = st.session_state.predictions_df.merge(
                        st.session_state.df_patches[needed_cols],
                        on='image_path',
                        how='inner'
                    )
                    preds_filt = merged[merged['parent_id'].str.contains(mapa_seleccionado.split('.')[0], na=False)]
                else:
                    st.error(f"El DataFrame de parches no tiene las columnas: {missing}")
                    preds_filt = pd.DataFrame()

                features = []
                for _, row in preds_filt.iterrows():
                    lon, lat = pixel_to_geo(row['min_x'], row['min_y'], gcps)
                    if lon is not None and lat is not None:
                        feature = {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [lon, lat]},
                            "properties": {
                                "image_path": row['image_path'],
                                "predicted_label": row['predicted_label'],
                                "probability": row['probability']
                            }
                        }
                        features.append(feature)

                if features:
                    geojson = {"type": "FeatureCollection", "features": features}
                    st.session_state.geojson_output = geojson
                    st.success(f"GeoJSON generado con {len(features)} puntos para el mapa {mapa_seleccionado}.")
                else:
                    st.warning("No se pudo transformar ningún punto. Verifica los GCPs o las predicciones.")
    else:
        st.info("Primero ejecuta una predicción (paso 4).")

    if st.button("🗑️ Limpiar archivos temporales"):
        if os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir)
        st.session_state.temp_dir = tempfile.mkdtemp()
        st.session_state.archivos_guardados = []
        st.session_state.map_images = None
        st.session_state.df_patches = None
        st.session_state.annotations_file = None
        st.session_state.classifier = None
        st.session_state.labels_map = None
        st.session_state.predictions_df = None
        st.session_state.geojson_output = None
        st.session_state.map_names = []
        st.success("Directorio temporal limpiado. Puedes empezar de nuevo.")

    st.markdown("---")
    st.info(f"📌 MapReader v{mapreader.__version__}")

# -------------------------------------------------------------------
# Área principal (dos columnas)
# -------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Parches generados")
    if st.session_state.df_patches is not None:
        st.dataframe(st.session_state.df_patches.head(20))
        if 'patch_id' in st.session_state.df_patches.columns:
            patch_ids = st.session_state.df_patches['patch_id'].tolist()
            selected_patch = st.selectbox("Selecciona un parche para ver", patch_ids[:50])
            if selected_patch:
                row = st.session_state.df_patches[st.session_state.df_patches['patch_id'] == selected_patch].iloc[0]
                image_path = row['image_path']
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    st.image(img, caption=f"Parche: {selected_patch}", use_container_width=True)
        else:
            st.warning("El DataFrame de parches no tiene la columna 'patch_id'.")
    else:
        st.info("Aún no hay parches. Procesa mapas en el panel izquierdo.")

with col2:
    st.subheader("📊 Predicciones")
    if st.session_state.predictions_df is not None:
        df_pred = st.session_state.predictions_df.copy()
        st.markdown("**Filtros**")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            if 'predicted_label' in df_pred.columns:
                labels = ['Todas'] + list(df_pred['predicted_label'].unique())
                label_filt = st.selectbox("Etiqueta", labels)
        with col_f2:
            if 'probability' in df_pred.columns:
                prob_thresh = st.slider("Umbral de confianza", 0.0, 1.0, 0.5)

        if label_filt != 'Todas':
            df_pred = df_pred[df_pred['predicted_label'] == label_filt]
        if 'probability' in df_pred.columns:
            df_pred = df_pred[df_pred['probability'] >= prob_thresh]

        st.write(f"**Mostrando {len(df_pred)} predicciones**")
        st.dataframe(df_pred.head(50))

        col_exp1, col_exp2, col_exp3 = st.columns(3)
        with col_exp1:
            csv = df_pred.to_csv(index=False).encode('utf-8')
            st.download_button("📥 CSV", csv, "predicciones.csv", "text/csv")
        with col_exp2:
            if st.session_state.geojson_output:
                geojson_str = json.dumps(st.session_state.geojson_output, indent=2)
                st.download_button("🌍 GeoJSON", geojson_str, "resultados.geojson", "application/json")
        with col_exp3:
            if st.session_state.geojson_output and len(st.session_state.geojson_output['features']) > 0:
                gdf = gpd.GeoDataFrame.from_features(st.session_state.geojson_output['features'])
                gdf.set_crs(epsg=4326, inplace=True)
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zf:
                    temp_shp = os.path.join(st.session_state.temp_dir, "temp.shp")
                    gdf.to_file(temp_shp, driver='ESRI Shapefile')
                    for ext in ['.shp', '.shx', '.dbf', '.prj']:
                        file_path = temp_shp.replace('.shp', ext)
                        if os.path.exists(file_path):
                            zf.write(file_path, arcname=f"resultados{ext}")
                zip_buffer.seek(0)
                st.download_button("📦 Shapefile", zip_buffer, "resultados_shapefile.zip", "application/zip")
    else:
        st.info("Aún no hay predicciones. Ejecuta el paso 4 en el panel izquierdo.")

# -------------------------------------------------------------------
# Visualización con Folium
# -------------------------------------------------------------------
if st.session_state.geojson_output:
    st.subheader("🗺️ Mapa interactivo de resultados georreferenciados")
    features = st.session_state.geojson_output['features']
    if features:
        lons = [f['geometry']['coordinates'][0] for f in features if f['geometry']['coordinates'][0] is not None]
        lats = [f['geometry']['coordinates'][1] for f in features if f['geometry']['coordinates'][1] is not None]
        if lons and lats:
            center = [np.mean(lats), np.mean(lons)]
            m = folium.Map(location=center, zoom_start=6)
            for f in features:
                lon, lat = f['geometry']['coordinates']
                if lon and lat:
                    label = f['properties']['predicted_label']
                    prob = f['properties']['probability']
                    popup = f"<b>Etiqueta:</b> {label}<br><b>Prob:</b> {prob:.2f}"
                    color = {'concesion':'red', 'limite':'blue', 'ferrocarril':'green', 'fondo':'gray'}.get(label, 'gray')
                    folium.CircleMarker(
                        [lat, lon],
                        radius=5,
                        popup=popup,
                        color=color,
                        fill=True,
                        fill_color=color
                    ).add_to(m)
            st_folium(m, width=800, height=500)
