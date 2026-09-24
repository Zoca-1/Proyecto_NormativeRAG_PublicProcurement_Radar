"""Dashboard Streamlit (ONLINE). Solo lee parquet/índice/GeoJSON precalculados.

No descarga, normaliza, reindexa ni simplifica geometrías al iniciar: si los
artefactos no existen, indica que se debe correr build_index.py primero.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src.hybrid_engine import HybridEngine, load_config
from src.map_visualizer import build_choropleth

st.set_page_config(page_title="Radar de Contrataciones Públicas", layout="wide")


@st.cache_resource
def get_engine() -> HybridEngine:
    return HybridEngine()


@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_geojson_cached(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


st.title("Radar de Contrataciones Públicas — Perú")

config = load_config()
base_dir = Path(__file__).resolve().parent
contracts_df = load_parquet(base_dir / config["paths"]["contracts_parquet"])
risk_df = load_parquet(base_dir / config["paths"]["risk_parquet"])
geo_metrics_df = load_parquet(base_dir / config["paths"]["geo_metrics_parquet"])

# El GeoJSON simplificado lo genera build_index.py (geo_loader.py); si aún no
# se corrió, se cae al original sin simplificar para no bloquear el mapa.
geojson_data = load_geojson_cached(base_dir / config["paths"]["geojson_simplificado_file"])
if geojson_data is None:
    geojson_data = load_geojson_cached(base_dir / config["geo"]["geojson_departamentos"])

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

tab_radar, tab_mapa, tab_riesgo, tab_tabla = st.tabs(
    ["Radar (preguntas)", "Mapa general", "Mapa de riesgo", "Tablas"]
)

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
    if geojson_data is None or geo_metrics_df.empty:
        st.info(
            "Falta el GeoJSON de departamentos o las métricas espaciales. "
            "Corre `build_index.py` (genera data/processed/metricas_departamento.parquet)."
        )
    else:
        metricas_disponibles = {
            "Monto adjudicado total (S/)": "monto_adjudicado_total",
            "Número de procesos": "numero_procesos",
            "Número de proveedores únicos": "numero_proveedores_unicos",
            "Número de entidades contratantes únicas": "numero_entidades_unicas",
        }
        metrica_label = st.selectbox("Métrica a visualizar", list(metricas_disponibles.keys()))
        value_column = metricas_disponibles[metrica_label]

        sin_datos = int((~geo_metrics_df["tiene_datos"]).sum())
        if sin_datos:
            st.caption(f"{sin_datos}/25 departamentos sin procesos en la muestra actual (mostrados en 0).")

        fig = build_choropleth(
            geo_metrics_df, geojson_data, config["geo"]["geojson_id_property"], "departamento",
            value_column, metrica_label,
            hover_columns=["numero_procesos", "numero_proveedores_unicos", "numero_entidades_unicas"],
            color_scale="Blues",
        )
        st.plotly_chart(fig, width='stretch')

with tab_riesgo:
    if geojson_data is None or risk_df.empty:
        st.info(
            "Falta el GeoJSON de departamentos o el indicador de riesgo. "
            "Corre `build_index.py` para generar data/processed/riesgo_departamento.parquet."
        )
    else:
        fig = build_choropleth(
            risk_df, geojson_data, config["geo"]["geojson_id_property"], "departamento",
            "tasa_pujador_unico", "Tasa pujador único",
            hover_columns=["total_contratos", "contratos_pujador_unico", "monto_total"],
            color_scale="Reds",
        )
        st.plotly_chart(fig, width='stretch')

with tab_tabla:
    st.subheader("Métricas por departamento")
    st.dataframe(geo_metrics_df, width='stretch')
    st.subheader("Indicador de riesgo (pujador único) por departamento")
    st.dataframe(risk_df, width='stretch')
