import streamlit as st
import os
import tempfile
import json
import re
import numpy as np
import pandas as pd
import folium
from PIL import Image
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Análisis de Tierras en Patagonia")
st.title("🗺️ Análisis de Asignación de Tierras en la Patagonia")

# -------------------------------------------------------------------
# Funciones reales
# -------------------------------------------------------------------
def crop_patches(image_path, patch_size, out_dir):
    """Recorta la imagen en parches reales y devuelve un DataFrame."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    patches = []
    pid = 0
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            x2 = min(x + patch_size, w)
            y2 = min(y + patch_size, h)
            crop = img.crop((x, y, x2, y2))
            fname = f"patch-{x}-{y}-{x2}-{y2}.png"
            crop.save(os.path.join(out_dir, fname))
            patches.append({
                "patch_id": f"patch_{pid:03d}",
                "parent_id": os.path.basename(image_path),
                "min_x": x, "min_y": y,
                "max_x": x2, "max_y": y2,
                "image_path": fname,
            })
            pid += 1
    return pd.DataFrame(patches)

def image_to_features(img_path, patch_size=200):
    """Convierte una imagen en un vector de features (histogramas RGB)."""
    img = Image.open(img_path).convert("RGB").resize((patch_size, patch_size))
    arr = np.array(img) / 255.0
    feats = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 1))
        feats.extend(hist / hist.sum())
        feats.append(arr[:, :, c].mean())
        feats.append(arr[:, :, c].std())
    return np.array(feats)

def parse_patch_coords(name):
    m = re.search(r'patch-(\d+)-(\d+)-(\d+)-(\d+)', str(name))
    return tuple(map(int, m.groups())) if m else None

def load_gcps(annotation):
    feats = annotation["items"][0]["body"]["features"]
    return [
        (tuple(f["properties"]["resourceCoords"]), tuple(f["geometry"]["coordinates"]))
        for f in feats
    ]

def fit_affine(gcps):
    A = np.array([[1, px, py] for (px, py), _ in gcps])
    LONS = np.array([lon for _, (lon, _) in gcps])
    LATS = np.array([lat for _, (_, lat) in gcps])
    return np.linalg.lstsq(A, LONS, rcond=None)[0], np.linalg.lstsq(A, LATS, rcond=None)[0]

def pixel_to_geo(x, y, cl, ca):
    return (cl[0] + cl[1]*x + cl[2]*y, ca[0] + ca[1]*x + ca[2]*y)

def build_geojson(df, cl, ca):
    features = []
    for _, row in df.iterrows():
        c = parse_patch_coords(row["image_path"])
        if c is None:
            continue
        x1, y1, x2, y2 = c
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        ring = [pixel_to_geo(x, y, cl, ca) for x, y in corners]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "label": str(row["predicted_label"]),
                "probability": float(row.get("probability", 0.0)),
                "patch": row["image_path"],
            },
        })
    return {"type": "FeatureCollection", "features": features}

# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------
for k, v in {
    "temp_dir": tempfile.mkdtemp(),
    "df_patches": None,
    "df_annotations": None,
    "model": None,
    "predictions_df": None,
    "geojson_output": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.header("1. Carga de mapa")
    uploaded = st.file_uploader("Subí tu mapa", type=["jpg", "jpeg", "png", "tiff", "tif"])
    patch_size = st.slider("Tamaño del parche (px)", 100, 500, 200)

    if uploaded and st.button("✂️ Generar parches"):
        img_path = os.path.join(st.session_state.temp_dir, uploaded.name)
        with open(img_path, "wb") as f:
            f.write(uploaded.getvalue())
        with st.spinner("Recortando parches reales..."):
            df = crop_patches(img_path, patch_size, st.session_state.temp_dir)
            st.session_state.df_patches = df
        st.success(f"✅ {len(df)} parches reales generados")

    st.divider()

    st.header("2. Anotaciones")
    csv_ann = st.file_uploader("CSV de anotaciones (image_path, label)", type=["csv"])
    if csv_ann:
        df = pd.read_csv(csv_ann)
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        st.session_state.df_annotations = df
        st.success(f"✅ {len(df)} anotaciones cargadas")

    st.divider()

    st.header("3. Entrenar")
    if st.button("🚀 Entrenar clasificador"):
        if st.session_state.df_patches is None or st.session_state.df_annotations is None:
            st.warning("Generá parches y cargá anotaciones primero.")
        else:
            with st.spinner("Extrayendo features y entrenando..."):
                X, y = [], []
                ann = st.session_state.df_annotations
                for _, row in ann.iterrows():
                    # Buscar el parche por nombre base (funciona con rutas absolutas o relativas)
                    base = os.path.basename(str(row["image_path"]))
                    p = os.path.join(st.session_state.temp_dir, base)
                    if os.path.exists(p):
                        X.append(image_to_features(p, patch_size))
                        y.append(row["label"])
                if len(X) < 5:
                    st.error(f"Muy pocas muestras encontradas ({len(X)}). Verificá las rutas del CSV.")
                else:
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                    clf.fit(np.array(X), np.array(y))
                    st.session_state.model = clf
                    st.success(f"✅ Modelo entrenado con {len(X)} muestras · clases: {sorted(set(y))}")

    st.divider()

    st.header("4. Predecir")
    if st.button("🔮 Predecir sobre todos los parches"):
        if st.session_state.model is None or st.session_state.df_patches is None:
            st.warning("Entrená un modelo y generá parches primero.")
        else:
            with st.spinner("Prediciendo..."):
                preds = []
                for _, row in st.session_state.df_patches.iterrows():
                    p = os.path.join(st.session_state.temp_dir, row["image_path"])
                    if os.path.exists(p):
                        f = image_to_features(p, patch_size).reshape(1, -1)
                        label = st.session_state.model.predict(f)[0]
                        prob = float(st.session_state.model.predict_proba(f).max())
                        preds.append({
                            "image_path": row["image_path"],
                            "predicted_label": label,
                            "probability": prob,
                        })
                df = pd.DataFrame(preds)
                st.session_state.predictions_df = df
            st.success(f"✅ {len(df)} predicciones generadas")

    st.divider()

    st.header("5. Georreferenciar")
    georef_file = st.file_uploader("Subí tu georef.json (Allmaps)", type=["json"])
    if georef_file:
        st.session_state.georef_json = json.load(georef_file)

    if st.button("🌍 Generar GeoJSON georreferenciado"):
        if st.session_state.predictions_df is None:
            st.warning("Generá predicciones primero.")
        elif "georef_json" not in st.session_state:
            st.error("Subí el georef.json de Allmaps primero.")
        else:
            with st.spinner("Aplicando transformación..."):
                gcps = load_gcps(st.session_state.georef_json)
                cl, ca = fit_affine(gcps)
                st.session_state.geojson_output = build_geojson(st.session_state.predictions_df, cl, ca)
            st.success(f"✅ GeoJSON generado con {len(gcps)} GCPs")

    st.divider()
    if st.button("🗑️ Limpiar todo"):
        st.session_state.df_patches = None
        st.session_state.df_annotations = None
        st.session_state.model = None
        st.session_state.predictions_df = None
        st.session_state.geojson_output = None
        st.success("Todo limpiado.")

# -------------------------------------------------------------------
# Área principal
# -------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Parches generados")
    if st.session_state.df_patches is not None:
        st.dataframe(st.session_state.df_patches.head(30), use_container_width=True)
    else:
        st.info("Subí un mapa y generá parches.")

with col2:
    st.subheader("📊 Predicciones")
    if st.session_state.predictions_df is not None:
        df = st.session_state.predictions_df
        label_filt = st.selectbox("Filtrar por etiqueta", ["Todas"] + sorted(df["predicted_label"].unique().tolist()))
        th = st.slider("Confianza mínima", 0.0, 1.0, 0.5, 0.05)
        dff = df if label_filt == "Todas" else df[df["predicted_label"] == label_filt]
        dff = dff[dff["probability"] >= th]
        st.dataframe(dff, use_container_width=True)
        st.download_button("📥 Descargar CSV", dff.to_csv(index=False).encode(), "predicciones.csv")
    else:
        st.info("Entrená y predecí para ver resultados.")

# -------------------------------------------------------------------
# Mapa
# -------------------------------------------------------------------
if st.session_state.get("geojson_output") is not None:
    st.subheader("🗺️ Mapa georreferenciado")
    gj = st.session_state.geojson_output
    lons = [c[0] for f in gj["features"] for c in f["geometry"]["coordinates"][0]]
    lats = [c[1] for f in gj["features"] for c in f["geometry"]["coordinates"][0]]
    center = [float(np.mean(lats)), float(np.mean(lons))]
    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    def style(f):
        c = {"pueblo": "red", "ferrocarril": "blue", "chacra": "green"}.get(f["properties"]["label"], "gray")
        return {"fillColor": c, "color": c, "weight": 0.5, "fillOpacity": 0.5}

    folium.GeoJson(
        gj, style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=["label", "probability", "patch"])
    ).add_to(m)
    st_folium(m, use_container_width=True, height=600)

    st.download_button(
        "🌍 Descargar GeoJSON",
        json.dumps(gj, indent=2),
        "clasificacion.geojson",
        "application/json",
    )
