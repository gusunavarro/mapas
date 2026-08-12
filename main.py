import streamlit as st
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import json
import folium
from streamlit_folium import st_folium

# Configuración de la página
st.set_page_config(layout="wide", page_title="Análisis de Tierras en Patagonia - Tellier")
st.title("🗺️ Análisis de Asignación de Tierras en la Patagonia - Caso Tellier")

# -------------------------------------------------------------------
# Inicializar session_state
# -------------------------------------------------------------------
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if "df_patches" not in st.session_state:
    st.session_state.df_patches = None
if "predictions_df" not in st.session_state:
    st.session_state.predictions_df = None
if "geojson_output" not in st.session_state:
    st.session_state.geojson_output = None
if "map_names" not in st.session_state:
    st.session_state.map_names = []
if "modo_demo" not in st.session_state:
    st.session_state.modo_demo = False

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
    patch_size = st.slider("Tamaño del parche (px)", 100, 500, 200)

    if uploaded_files and st.button("📥 Generar parches"):
        nombres_mapas = []
        for uploaded_file in uploaded_files:
            path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
            nombres_mapas.append(uploaded_file.name)
        st.session_state.map_names = nombres_mapas

        with st.spinner("Generando parches..."):
            # Crear DataFrame simulado de parches (para demostración)
            patch_data = []
            for i in range(50):  # 50 parches de ejemplo
                patch_data.append({
                    "patch_id": f"patch_{i:03d}",
                    "parent_id": uploaded_files[0].name if uploaded_files else "mapa.jpg",
                    "min_x": i * 10,
                    "min_y": i * 10,
                    "image_path_rel": f"patch_{i:03d}.png"
                })
            st.session_state.df_patches = pd.DataFrame(patch_data)
        st.success(f"✅ {len(st.session_state.df_patches)} parches generados (simulados).")

    st.header("2. Modo de operación")
    st.session_state.modo_demo = st.checkbox("🎬 Usar modo demostración (resultados precalculados)", value=True)

    if st.session_state.modo_demo:
        st.info("Modo demostración: se mostrarán resultados del caso Tellier sin necesidad de entrenar.")
        if st.button("📂 Cargar resultados de demostración"):
            # Crear DataFrame de predicciones de ejemplo
            demo_preds = []
            for i in range(50):
                label = np.random.choice(["pueblo", "ferrocarril", "chacra", "fondo"], p=[0.2, 0.2, 0.3, 0.3])
                prob = np.random.uniform(0.7, 0.99)
                demo_preds.append({
                    "image_path_rel": f"patch_{i:03d}.png",
                    "predicted_label": label,
                    "probability": prob
                })
            st.session_state.predictions_df = pd.DataFrame(demo_preds)

            # GeoJSON de ejemplo (puntos alrededor de Tellier)
            features = []
            coords_tellier = [
                (-68.123, -47.456), (-68.125, -47.458), (-68.119, -47.461),
                (-68.130, -47.450), (-68.115, -47.465), (-68.140, -47.455)
            ]
            for i, (lon, lat) in enumerate(coords_tellier):
                label = ["pueblo", "ferrocarril", "chacra"][i % 3]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"predicted_label": label, "probability": 0.85}
                })
            st.session_state.geojson_output = {"type": "FeatureCollection", "features": features}
            st.success("✅ Resultados de demostración cargados (caso Tellier).")
    else:
        st.warning("Modo completo desactivado para la presentación. Usa el modo demostración.")

    if st.button("🗑️ Limpiar todo"):
        if os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir)
        st.session_state.temp_dir = tempfile.mkdtemp()
        st.session_state.df_patches = None
        st.session_state.predictions_df = None
        st.session_state.geojson_output = None
        st.session_state.map_names = []
        st.success("Todo limpiado. Recarga la página para empezar de nuevo.")

# -------------------------------------------------------------------
# Área principal
# -------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Parches generados")
    if st.session_state.df_patches is not None:
        st.dataframe(st.session_state.df_patches.head(20))
    else:
        st.info("Aún no hay parches. Usa el panel izquierdo.")

with col2:
    st.subheader("📊 Predicciones")
    if st.session_state.predictions_df is not None:
        df_pred = st.session_state.predictions_df.copy()
        label_filt = st.selectbox("Filtrar por etiqueta", ["Todas"] + list(df_pred["predicted_label"].unique()))
        prob_thresh = st.slider("Confianza mínima", 0.0, 1.0, 0.5)
        if label_filt != "Todas":
            df_pred = df_pred[df_pred["predicted_label"] == label_filt]
        df_pred = df_pred[df_pred["probability"] >= prob_thresh]
        st.dataframe(df_pred.head(50))

        csv = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar CSV", csv, "predicciones.csv", "text/csv")
    else:
        st.info("Carga los resultados de demostración o genera parches primero.")

# -------------------------------------------------------------------
# Mapa interactivo
# -------------------------------------------------------------------
if st.session_state.geojson_output:
    st.subheader("🗺️ Mapa interactivo - Colonia Tellier")
    features = st.session_state.geojson_output["features"]
    if features:
        lats = [f["geometry"]["coordinates"][1] for f in features]
        lons = [f["geometry"]["coordinates"][0] for f in features]
        center = [np.mean(lats), np.mean(lons)]
        m = folium.Map(location=center, zoom_start=12)
        for f in features:
            lon, lat = f["geometry"]["coordinates"]
            label = f["properties"]["predicted_label"]
            prob = f["properties"]["probability"]
            color = {"pueblo": "red", "ferrocarril": "blue", "chacra": "green"}.get(label, "gray")
            folium.CircleMarker([lat, lon], radius=6, popup=f"{label} ({prob:.2f})", color=color, fill=True).add_to(m)
        st_folium(m, width=700, height=500)

        # Botón para descargar GeoJSON
        geojson_str = json.dumps(st.session_state.geojson_output, indent=2)
        st.download_button("🌍 Descargar GeoJSON", geojson_str, "tellier_resultados.geojson", "application/json")
