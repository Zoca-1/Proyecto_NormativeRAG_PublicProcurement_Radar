"""Script OFFLINE: geo -> fetch OECE -> normalización -> riesgo -> métricas
espaciales -> índice híbrido.

Idempotente y reanudable: el fetch retoma desde _fetch_state.json, y cada
etapa posterior se puede re-ejecutar sin duplicar filas (dedupe por ocid).
La verificación geográfica (paso 0) no depende de haber descargado datos de
contrataciones: corre siempre, solo con el GeoJSON.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import pandas as pd

from src.geo_loader import load_and_prepare_geojson
from src.hybrid_engine import load_config
from src.hybrid_indexer import HybridIndex, build_embedding_provider, build_texto_para_embedding
from src.oece_fetcher import fetch_releases
from src.risk_analyzer import compute_risk_by_categoria, compute_risk_indicators
from src.spatial_analysis import compute_departamento_metrics, join_geo_metrics
from src.validator_normalizer import load_and_normalize, save_rejected


def render_geo_inicial_report(geo_issues: list[str], ubigeo_table: pd.DataFrame,
                               geo_metrics_df: pd.DataFrame, n_contratos: int) -> str:
    lines = [
        "# Reporte Inicial — Radar Geográfico (Tarea 2)",
        "",
        f"Generado: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. Verificación del GeoJSON de departamentos",
        "",
    ]
    if geo_issues:
        lines.append(f"⚠️ Se encontraron {len(geo_issues)} problemas (ver detalle abajo); revisar antes de usar el mapa en producción:")
        lines.extend(f"- {issue}" for issue in geo_issues)
    else:
        lines.append(
            "Sin problemas: FeatureCollection válido, proyección WGS84/CRS84, "
            "25 departamentos del catálogo presentes 1:1, todas las geometrías "
            "son Polygon/MultiPolygon válidas según shapely. Ver `src/test_geo.py` "
            "para la verificación programática completa (corre independiente de este script)."
        )
    lines.append("")
    lines.append(
        f"Tabla ubigeo(INEI)<->departamento extraída de las {len(ubigeo_table)} "
        "features del propio GeoJSON (columna `ubigeo_property` de config.yaml -> geo)."
    )
    lines.append("")

    lines.append("## 2. Integración con datos de contratación regional")
    lines.append("")
    if n_contratos == 0:
        lines.append(
            "Aún no hay contratos normalizados (`data/processed/contratos.parquet` vacío o "
            "inexistente); la tabla de métricas por departamento existe pero con todas las "
            "métricas en 0 (`tiene_datos=False` para las 25 regiones). Corre `build_index.py` "
            "después de que `oece_fetcher.py` haya descargado releases."
        )
    else:
        con_datos = int(geo_metrics_df["tiene_datos"].sum())
        lines.append(
            f"{n_contratos} contratos normalizados agregados a nivel de departamento. "
            f"{con_datos}/25 departamentos tienen al menos un proceso registrado en esta muestra."
        )
        lines.append("")
        top5 = geo_metrics_df.sort_values("monto_adjudicado_total", ascending=False).head(5)
        lines.append("Top 5 departamentos por monto adjudicado en esta muestra:")
        lines.append("")
        lines.append("| Departamento | Monto adjudicado | N° procesos | N° proveedores únicos | N° entidades únicas |")
        lines.append("|---|---|---|---|---|")
        for _, row in top5.iterrows():
            lines.append(
                f"| {row['departamento']} | S/ {row['monto_adjudicado_total']:,.2f} | "
                f"{int(row['numero_procesos'])} | {int(row['numero_proveedores_unicos'])} | "
                f"{int(row['numero_entidades_unicas'])} |"
            )
    lines.append("")

    lines.append("## 3. Modelo de relación Ubigeo/Departamento <-> métricas")
    lines.append("")
    lines.append(
        "`src/spatial_analysis.py::compute_departamento_metrics` agrega "
        "`contratos.parquet` por `departamento` (monto adjudicado total, número de "
        "procesos, proveedores únicos vía `nunique` sobre `proveedor`, entidades "
        "únicas vía `nunique` sobre `comprador`). "
        "`join_geo_metrics` hace un LEFT JOIN de la tabla ubigeo (25 filas, siempre "
        "completa, viene del GeoJSON) con esas métricas, dejando explícito con "
        "`tiene_datos=False` cualquier departamento sin procesos en la muestra "
        "actual, en vez de que desaparezca silenciosamente del mapa."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el pipeline de datos y el índice híbrido de Tarea 2.")
    parser.add_argument("--refresh", action="store_true", help="Reinicia el estado de paginación y re-descarga todo.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config = load_config()

    raw_releases_path = base_dir / config["paths"]["raw_releases_file"]
    state_path = base_dir / config["paths"]["fetch_state_file"]
    contracts_path = base_dir / config["paths"]["contracts_parquet"]
    risk_path = base_dir / config["paths"]["risk_parquet"]
    geo_metrics_path = base_dir / config["paths"]["geo_metrics_parquet"]
    geojson_simplificado_path = base_dir / config["paths"]["geojson_simplificado_file"]
    index_path = base_dir / config["paths"]["index_file"]
    rejected_path = base_dir / config["normalizacion"]["rejected_log_file"]

    if args.refresh:
        raw_releases_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    print("[0/6] Verificando y preparando GeoJSON de departamentos...")
    geojson, ubigeo_table, geo_issues = load_and_prepare_geojson(config, base_dir)
    geojson_simplificado_path.parent.mkdir(parents=True, exist_ok=True)
    with geojson_simplificado_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    if geo_issues:
        print(f"    {len(geo_issues)} problema(s) encontrados (ver docs/reporte_geo_inicial.md)")
    else:
        print("    OK: 25 departamentos verificados, WGS84, geometrías válidas")

    new_releases = fetch_releases(config["oece"], raw_releases_path, state_path)
    print(f"[1/6] Fetch OECE: {new_releases} releases nuevos descargados")

    df = pd.DataFrame()
    if raw_releases_path.exists():
        df, rejected = load_and_normalize(raw_releases_path, config["departamentos"], config["normalizacion"])
        save_rejected(rejected, rejected_path)
        contracts_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(contracts_path, index=False)
        print(f"[2/6] Normalización: {len(df)} contratos válidos, {len(rejected)} filas rechazadas")
    else:
        print("[2/6] Normalización: sin datos crudos todavía, se omite")

    if not df.empty:
        risk_df = compute_risk_indicators(df, config["riesgo"])
        risk_df.to_parquet(risk_path, index=False)
        compute_risk_by_categoria(df, config["riesgo"])
        print(f"[3/6] Indicador de riesgo: {len(risk_df)} departamentos con métricas de pujador único")
    else:
        print("[3/6] Indicador de riesgo: sin datos, se omite")

    print("[4/6] Métricas espaciales por departamento...")
    dept_metrics = compute_departamento_metrics(df)
    geo_metrics_df = join_geo_metrics(ubigeo_table, dept_metrics)
    geo_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    geo_metrics_df.to_parquet(geo_metrics_path, index=False)
    con_datos = int(geo_metrics_df["tiene_datos"].sum())
    print(f"    {con_datos}/25 departamentos con datos en esta muestra -> {geo_metrics_path}")

    if not df.empty:
        index = HybridIndex.load(index_path)
        existing_ocids = set(index.metadata["ocid"]) if not index.metadata.empty else set()
        pending_df = df[~df["ocid"].isin(existing_ocids)].reset_index(drop=True)

        if pending_df.empty:
            print("[5/6] Embeddings: ok (índice ya actualizado)")
        else:
            provider = build_embedding_provider(config["embeddings"])
            campos = config["campos_texto_para_embedding"]
            textos = pending_df.apply(lambda row: build_texto_para_embedding(row, campos), axis=1).tolist()
            vectors = provider.embed(textos)
            added = index.add(pending_df, vectors)
            index.save(index_path)
            print(f"[5/6] Embeddings: {added} vectores nuevos agregados al índice híbrido")
        print(f"[6/6] Índice final: {len(index)} contratos indexados en {index_path}")
    else:
        print("[5/6] Embeddings: sin datos, se omite")
        print("[6/6] Índice final: sin datos, se omite")

    report_md = render_geo_inicial_report(geo_issues, ubigeo_table, geo_metrics_df, len(df))
    report_path = base_dir / config["paths"]["geo_inicial_report_file"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Reporte geográfico inicial guardado en {report_path}")


if __name__ == "__main__":
    main()
