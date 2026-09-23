"""Construcción de figuras de mapa (plotly). Sin streamlit: retorna objetos
plotly.graph_objects.Figure que la UI (app.py) solo tiene que renderizar con
st.plotly_chart(). Mantiene la lógica de mapa reutilizable y testeable sin
levantar la app.
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go


def build_choropleth(df, geojson: dict, id_property: str, location_column: str, value_column: str,
                      label: str, hover_columns: list[str], color_scale: str = "Reds") -> go.Figure:
    """Mapa coroplético genérico por departamento. value_column determina qué
    métrica se colorea (monto, número de procesos, tasa de riesgo, etc.), así
    que la misma función sirve para cualquier tab del dashboard.
    """
    fig = px.choropleth(
        df,
        geojson=geojson,
        locations=location_column,
        featureidkey=f"properties.{id_property}",
        color=value_column,
        color_continuous_scale=color_scale,
        hover_data=hover_columns,
        labels={value_column: label},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
    return fig
