"""Dashboard Streamlit (ONLINE). Solo lee parquet/índice precalculados.

No descarga, normaliza ni reindexa nada al iniciar: si los artefactos no
existen, indica que se debe correr build_index.py primero.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.hybrid_engine import HybridEngine, load_config

st.set_page_config(page_title="Radar de Contrataciones Públicas", layout="wide")


@st.cache_resource
def get_engine() -> HybridEngine:
    return HybridEngine()


@st.cache_data
def load_contracts(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_risk(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


st.title("Radar de Contrataciones Públicas — Perú")

config = load_config()
base_dir = Path(__file__).resolve().parent
contracts_df = load_contracts(base_dir / config["paths"]["contracts_parquet"])
risk_df = load_risk(base_dir / config["paths"]["risk_parquet"])

if contracts_df.empty:
    st.warning("No hay datos procesados. Ejecuta `python build_index.py` antes de iniciar la aplicación.")
    st.stop()

st.sidebar.header("Filtros")
departamentos = ["(todos)"] + sorted(contracts_df["departamento"].dropna().unique().tolist())
categorias = ["(todas)"] + sorted(contracts_df["categoria"].dropna().unique().tolist())

departamento_sel = st.sidebar.selectbox("Departamento", departamentos)
categoria_sel = st.sidebar.selectbox("Categoría", categorias)
monto_min, monto_max = st.sidebar.slider(
    "Rango de monto (S/)",
    float(contracts_df["monto"].min()), float(contracts_df["monto"].max()),
    (float(contracts_df["monto"].min()), float(contracts_df["monto"].max())),
)

tab_radar, tab_mapa, tab_riesgo = st.tabs(["Radar (preguntas)", "Mapa de riesgo", "Tabla de riesgo"])

with tab_radar:
    question = st.text_input("Pregunta sobre las contrataciones públicas")
    if st.button("Consultar") and question.strip():
        filters = {"monto_min": monto_min, "monto_max": monto_max}
        if departamento_sel != "(todos)":
            filters["departamento"] = departamento_sel
        if categoria_sel != "(todas)":
            filters["categoria"] = categoria_sel

        try:
            engine = get_engine()
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo cargar el motor: {exc}")
            st.stop()

        with st.spinner("Buscando contratos relevantes..."):
            result = engine.query(question, filters)

        if result["error"]:
            st.error(result["error"])
        else:
            if result["abstained"]:
                st.warning(result["answer"])
            else:
                st.markdown(result["answer"])

            st.subheader("Contratos citados")
            for source in result["sources"]:
                st.markdown(
                    f"- **OCID {source['ocid']}** — {source['departamento']} · S/ {source['monto']:,.2f} "
                    f"· {source['categoria']} · similitud {source['similitud']:.3f}"
                )

            col1, col2, col3 = st.columns(3)
            col1.metric("Costo (USD)", f"${result['cost_usd']:.5f}")
            col2.metric("Tokens (in/out)", f"{result['tokens_in']}/{result['tokens_out']}")
            col3.metric("Latencia (s)", f"{result['latency_sec']:.2f}")

with tab_mapa:
    geojson_path = base_dir / config["geo"]["geojson_departamentos"]
    if not geojson_path.exists() or risk_df.empty:
        st.info(
            "Falta el GeoJSON de departamentos o el indicador de riesgo. "
            "Coloca el archivo en la ruta de config.yaml -> geo.geojson_departamentos y corre build_index.py."
        )
    else:
        with geojson_path.open("r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        fig = px.choropleth(
            risk_df,
            geojson=geojson_data,
            locations="departamento",
            featureidkey=f"properties.{config['geo']['geojson_id_property']}",
            color="tasa_pujador_unico",
            color_continuous_scale="Reds",
            hover_data=["total_contratos", "contratos_pujador_unico", "monto_total"],
            labels={"tasa_pujador_unico": "Tasa pujador único"},
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)

with tab_riesgo:
    st.dataframe(risk_df, use_container_width=True)
