"""Indicador de riesgo por pujador único, agregado por departamento y categoría."""
from __future__ import annotations

import pandas as pd


def compute_risk_indicators(df: pd.DataFrame, riesgo_cfg: dict) -> pd.DataFrame:
    min_postores_alerta = riesgo_cfg["min_postores_alerta"]

    working = df.copy()
    working["es_pujador_unico"] = working["numero_postores"].fillna(0) <= min_postores_alerta

    grouped = working.groupby("departamento").agg(
        total_contratos=("ocid", "count"),
        contratos_pujador_unico=("es_pujador_unico", "sum"),
        monto_total=("monto", "sum"),
    ).reset_index()

    grouped["tasa_pujador_unico"] = (
        grouped["contratos_pujador_unico"] / grouped["total_contratos"]
    ).round(4)
    return grouped.sort_values("tasa_pujador_unico", ascending=False)


def compute_risk_by_categoria(df: pd.DataFrame, riesgo_cfg: dict) -> pd.DataFrame:
    min_postores_alerta = riesgo_cfg["min_postores_alerta"]

    working = df.copy()
    working["es_pujador_unico"] = working["numero_postores"].fillna(0) <= min_postores_alerta

    grouped = working.groupby(["departamento", "categoria"]).agg(
        total_contratos=("ocid", "count"),
        contratos_pujador_unico=("es_pujador_unico", "sum"),
    ).reset_index()
    grouped["tasa_pujador_unico"] = (
        grouped["contratos_pujador_unico"] / grouped["total_contratos"]
    ).round(4)
    return grouped.sort_values("tasa_pujador_unico", ascending=False)
