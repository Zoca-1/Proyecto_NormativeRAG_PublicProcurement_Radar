"""UI Streamlit (ONLINE). Solo lee el índice precalculado y llama a rag_engine.

No reconstruye ni re-extrae nada al iniciar: si el índice no existe, muestra
un mensaje indicando que se debe correr build_index.py primero.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.rag_engine import RagEngine, load_config

st.set_page_config(page_title="RAG Normativo - Contrataciones Públicas", layout="wide")


@st.cache_resource
def get_engine() -> RagEngine:
    return RagEngine()


st.title("Asistente Normativo — Ley N.° 32069 y D.S. N.° 001-2026-EF")

config = load_config()
try:
    engine = get_engine()
except Exception as exc:  # noqa: BLE001
    st.error(f"No se pudo cargar el motor: {exc}")
    st.info("Ejecuta `python build_index.py` antes de iniciar la aplicación.")
    st.stop()

if len(engine.store) == 0:
    st.warning("El índice está vacío. Ejecuta `python build_index.py` antes de consultar.")
    st.stop()

documento_ids = ["(todos)"] + [d["id"] for d in config["documents"]]
documento_sel = st.sidebar.selectbox("Filtrar por documento", documento_ids)

question = st.text_input("Pregunta sobre la normativa de contrataciones públicas")

if st.button("Consultar") and question.strip():
    filters = None if documento_sel == "(todos)" else {"documento_id": documento_sel}
    with st.spinner("Buscando en la normativa..."):
        result = engine.query(question, filters)

    if result["error"]:
        st.error(result["error"])
    else:
        if result["abstained"]:
            st.warning(result["answer"])
        else:
            st.markdown(result["answer"])
            if result.get("citations_verified") is False:
                st.caption("⚠️ No se detectó una cita explícita de artículo/página en el texto de la respuesta.")

        st.subheader("Fuentes citadas")
        for source in result["sources"]:
            detalle = " · ".join(
                filter(None, [source.get("titulo"), source.get("articulo"),
                              f"numeral {source['numeral']}" if source.get("numeral") else None,
                              f"inciso {source['inciso']}" if source.get("inciso") else None])
            )
            st.markdown(
                f"- **{source['documento']}** (v.{source['version']}) — pág. {source['page_number']}"
                + (f" — {detalle}" if detalle else "")
                + f" · similitud {source['similitud']:.3f}"
            )

        col1, col2, col3 = st.columns(3)
        col1.metric("Costo (USD)", f"${result['cost_usd']:.5f}")
        col2.metric("Tokens (in/out)", f"{result['tokens_in']}/{result['tokens_out']}")
        col3.metric("Latencia (s)", f"{result['latency_sec']:.2f}")
