"""Carga, validación y preparación del GeoJSON de departamentos del Perú.

Responsabilidades:
  - Cargar el archivo y validar que sea un FeatureCollection en WGS84
    (EPSG:4326), que traiga exactamente las 25 regiones del catálogo
    canónico (config.yaml -> departamentos.lista) y que cada geometría sea
    válida (Polygon/MultiPolygon).
  - Extraer la tabla departamento -> ubigeo a partir de las propiedades del
    propio GeoJSON (evita mantener un catálogo ubigeo por separado que se
    pueda desincronizar).
  - Simplificar geometrías (Douglas-Peucker vía shapely) para acelerar el
    renderizado del mapa, solo si aporta una reducción significativa de
    puntos.

RFC 7946 (el estándar GeoJSON) exige que todas las coordenadas estén en
WGS84 (CRS84, lon/lat); un miembro "crs" que declare otra cosa es un caso a
alertar explícitamente, no a asumir silenciosamente como compatible.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
from shapely.geometry import mapping, shape

_WGS84_CRS_NAMES = {
    "urn:ogc:def:crs:ogc:1.3:crs84",
    "urn:ogc:def:crs:epsg::4326",
    "epsg:4326",
}


class GeoValidationError(ValueError):
    """El GeoJSON no cumple los requisitos mínimos para usarse en el mapa."""


def load_geojson(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró '{path}'. Coloca el GeoJSON de departamentos en la ruta "
            "declarada en config.yaml -> geo.geojson_departamentos."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_geojson(geojson: dict, departamentos_cfg: dict, geo_cfg: dict) -> list[str]:
    """Retorna una lista de problemas encontrados (vacía si todo está bien).
    No lanza excepción: quien llama decide si un problema es bloqueante.
    """
    issues: list[str] = []
    id_property = geo_cfg["geojson_id_property"]
    catalogo = set(departamentos_cfg["lista"])

    if geojson.get("type") != "FeatureCollection":
        issues.append(f"type esperado 'FeatureCollection', encontrado '{geojson.get('type')}'")

    crs = geojson.get("crs")
    if crs is not None:
        crs_name = (crs.get("properties", {}).get("name") or "").lower()
        if crs_name not in _WGS84_CRS_NAMES:
            issues.append(f"crs declarado no es WGS84/CRS84 (encontrado: '{crs_name}')")
    # Si no hay miembro "crs", RFC 7946 asume WGS84 por defecto: no es un problema.

    features = geojson.get("features", [])
    if len(features) != len(catalogo):
        issues.append(f"se esperaban {len(catalogo)} features (departamentos), se encontraron {len(features)}")

    nombres_encontrados = []
    for i, feature in enumerate(features):
        props = feature.get("properties", {})
        nombre = props.get(id_property)
        if not nombre:
            issues.append(f"feature #{i} no tiene la propiedad '{id_property}'")
            continue
        nombres_encontrados.append(nombre.strip().upper())

        geometry = feature.get("geometry", {})
        if geometry.get("type") not in ("Polygon", "MultiPolygon"):
            issues.append(f"feature '{nombre}' tiene geometry.type inesperado: '{geometry.get('type')}'")
            continue
        try:
            geom = shape(geometry)
            if not geom.is_valid:
                issues.append(f"feature '{nombre}' tiene una geometría inválida (shapely.is_valid=False)")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"feature '{nombre}': error al construir la geometría ({exc})")

    duplicados = {n for n in nombres_encontrados if nombres_encontrados.count(n) > 1}
    if duplicados:
        issues.append(f"departamentos duplicados en el GeoJSON: {sorted(duplicados)}")

    faltantes = catalogo - set(nombres_encontrados)
    if faltantes:
        issues.append(f"departamentos del catálogo ausentes en el GeoJSON: {sorted(faltantes)}")

    inesperados = set(nombres_encontrados) - catalogo
    if inesperados:
        issues.append(f"nombres en el GeoJSON que no están en el catálogo de 25 departamentos: {sorted(inesperados)}")

    return issues


def extract_departamento_ubigeo(geojson: dict, id_property: str, ubigeo_property: str | None) -> pd.DataFrame:
    """Tabla departamento -> ubigeo, leída directamente de las propiedades
    del GeoJSON (no de un catálogo externo, para no desincronizarse)."""
    rows = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        rows.append({
            "departamento": (props.get(id_property) or "").strip().upper(),
            "ubigeo_departamento": props.get(ubigeo_property) if ubigeo_property else None,
        })
    return pd.DataFrame(rows)


def simplify_geometries(geojson: dict, tolerance: float, min_points_to_simplify: int) -> dict:
    """Simplifica cada geometría con Douglas-Peucker (shapely), preservando
    topología, solo si tiene más de min_points_to_simplify puntos. Retorna
    un GeoJSON nuevo (no muta el original).
    """
    simplified = copy.deepcopy(geojson)
    for feature in simplified.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        geom = shape(geometry)
        if geom.geom_type == "Polygon":
            point_count = len(geom.exterior.coords)
        else:
            point_count = sum(len(poly.exterior.coords) for poly in geom.geoms)
        if point_count < min_points_to_simplify:
            continue
        simplified_geom = geom.simplify(tolerance, preserve_topology=True)
        feature["geometry"] = mapping(simplified_geom)
    return simplified


def load_and_prepare_geojson(config: dict, base_dir: Path) -> tuple[dict, pd.DataFrame, list[str]]:
    """Orquesta carga -> validación -> extracción ubigeo -> simplificación.
    Retorna (geojson_listo_para_mapa, tabla_departamento_ubigeo, issues).
    Los issues se retornan (no se lanzan) para que build_index.py y
    test_geo.py decidan cómo reportarlos.
    """
    geo_cfg = config["geo"]
    path = base_dir / geo_cfg["geojson_departamentos"]

    geojson = load_geojson(path)
    issues = validate_geojson(geojson, config["departamentos"], geo_cfg)
    ubigeo_table = extract_departamento_ubigeo(geojson, geo_cfg["geojson_id_property"], geo_cfg.get("ubigeo_property"))

    simplification_cfg = geo_cfg.get("simplification", {})
    if simplification_cfg.get("enabled", False):
        geojson = simplify_geometries(
            geojson, simplification_cfg["tolerance"], simplification_cfg["min_points_to_simplify"],
        )

    return geojson, ubigeo_table, issues
