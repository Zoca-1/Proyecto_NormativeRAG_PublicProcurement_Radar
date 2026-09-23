"""Relación Ubigeo/Departamento <-> métricas de contratación pública.

Agrega contratos.parquet a nivel de departamento (monto adjudicado, número de
procesos, proveedores y entidades contratantes únicos) y la une con la tabla
departamento->ubigeo extraída del GeoJSON (geo_loader.extract_departamento_ubigeo),
para tener un único DataFrame lat/lon-agnóstico listo para el mapa
(map_visualizer.py) y para inspección tabular.

No duplica risk_analyzer.py: ese módulo calcula el indicador específico de
riesgo por pujador único; este módulo calcula las métricas generales de
volumen/actividad por departamento. build_index.py combina ambos.
"""
from __future__ import annotations

import pandas as pd


def compute_departamento_metrics(contracts_df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por departamento con las métricas generales de contratación."""
    if contracts_df.empty:
        return pd.DataFrame(columns=[
            "departamento", "monto_adjudicado_total", "numero_procesos",
            "numero_proveedores_unicos", "numero_entidades_unicas",
        ])

    grouped = contracts_df.groupby("departamento").agg(
        monto_adjudicado_total=("monto", "sum"),
        numero_procesos=("ocid", "count"),
        numero_proveedores_unicos=("proveedor", "nunique"),
        numero_entidades_unicas=("comprador", "nunique"),
    ).reset_index()
    return grouped.sort_values("monto_adjudicado_total", ascending=False)


def join_geo_metrics(ubigeo_table: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Une la tabla departamento->ubigeo (25 filas, siempre completa, viene del
    GeoJSON) con las métricas (pueden faltar departamentos sin contratos
    registrados todavía). Un outer join con indicador deja explícitos los
    departamentos sin datos en vez de que desaparezcan silenciosamente del mapa.
    """
    merged = ubigeo_table.merge(metrics_df, on="departamento", how="left", indicator=True)
    metric_cols = [c for c in metrics_df.columns if c != "departamento"]
    merged[metric_cols] = merged[metric_cols].fillna(0)
    merged["tiene_datos"] = merged["_merge"] == "both"
    return merged.drop(columns="_merge")
