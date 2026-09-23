"""Verificación de integridad del GeoJSON de departamentos y de la
asignación correcta de las 25 regiones. Script standalone (no pytest) para
poder correrlo igual que source_check.py en Tarea 1: `python src/test_geo.py`
desde tarea2_radar/, o `python -m tarea2_radar.src.test_geo`.

No requiere datos de contrataciones ni el LLM: solo el GeoJSON y config.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from shapely.geometry import shape

from src.geo_loader import extract_departamento_ubigeo, load_geojson, validate_geojson


def _check(label: str, condition: bool, detail: str = "") -> bool:
    status = "OK" if condition else "FALLO"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def main() -> int:
    base_dir = Path(__file__).resolve().parent.parent
    with (base_dir / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    geo_cfg = config["geo"]
    departamentos_cfg = config["departamentos"]
    geojson_path = base_dir / geo_cfg["geojson_departamentos"]

    all_ok = True

    all_ok &= _check(f"El archivo existe ({geojson_path})", geojson_path.exists())
    if not geojson_path.exists():
        print("\nNo se puede continuar sin el archivo. Abortando.")
        return 1

    geojson = load_geojson(geojson_path)
    all_ok &= _check("type == 'FeatureCollection'", geojson.get("type") == "FeatureCollection")

    issues = validate_geojson(geojson, departamentos_cfg, geo_cfg)
    all_ok &= _check("Sin problemas de validate_geojson (CRS, conteo, geometrías, catálogo)", not issues)
    for issue in issues:
        print(f"    - {issue}")

    catalogo = set(departamentos_cfg["lista"])
    id_property = geo_cfg["geojson_id_property"]
    nombres = {
        (f.get("properties", {}).get(id_property) or "").strip().upper()
        for f in geojson.get("features", [])
    }
    all_ok &= _check("Exactamente 25 departamentos en el catálogo", len(catalogo) == 25, f"catálogo tiene {len(catalogo)}")
    all_ok &= _check("Las 25 regiones del GeoJSON coinciden 1:1 con el catálogo", nombres == catalogo,
                      f"diff: catálogo-geojson={catalogo - nombres}, geojson-catálogo={nombres - catalogo}")

    geometry_types = {f.get("geometry", {}).get("type") for f in geojson.get("features", [])}
    all_ok &= _check("Todas las geometrías son Polygon o MultiPolygon", geometry_types <= {"Polygon", "MultiPolygon"},
                      f"tipos encontrados: {geometry_types}")

    validas = all(shape(f["geometry"]).is_valid for f in geojson.get("features", []) if f.get("geometry"))
    all_ok &= _check("Todas las geometrías son válidas según shapely (is_valid)", validas)

    ubigeo_property = geo_cfg.get("ubigeo_property")
    if ubigeo_property:
        ubigeo_table = extract_departamento_ubigeo(geojson, id_property, ubigeo_property)
        codigos = ubigeo_table["ubigeo_departamento"].dropna().tolist()
        all_ok &= _check(
            "Códigos ubigeo de departamento presentes y únicos (25 códigos distintos)",
            len(set(codigos)) == 25 == len(codigos),
            f"{len(codigos)} códigos, {len(set(codigos))} únicos",
        )
    else:
        print("[SKIP] ubigeo_property no configurado en config.yaml -> geo; se omite la verificación de ubigeo.")

    print()
    if all_ok:
        print("RESULTADO: todas las verificaciones pasaron.")
        return 0
    print("RESULTADO: hay verificaciones fallidas — revisar antes de usar el GeoJSON en el mapa.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
