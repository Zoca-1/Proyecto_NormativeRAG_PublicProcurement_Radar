"""Evaluación del Radar de Contrataciones (Tarea 2), costo-cero (no llama al LLM):

  1. Retrieval híbrido: para cada pregunta con filtros estructurados, mide
     candidatos tras el filtro y similitud top-1/top-k contra el umbral de
     abstención configurado.
  2. Cobertura de los 25 departamentos: en contratos.parquet, en el índice
     híbrido, y en las respuestas de retrieval (que cada filtro devuelva
     candidatos reales, no solo que el departamento "exista" en el catálogo).
  3. Precisión/consistencia del índice de riesgo (risk_analyzer.py): no hay
     etiquetas externas de "riesgo real" contra las cuales medir precisión
     en el sentido de clasificación supervisada, así que esto verifica
     CORRECTITUD ARITMÉTICA — recalcula el indicador desde contratos.parquet
     y lo compara contra el riesgo_departamento.parquet persistido, fila por
     fila. También reporta el hallazgo de ocids duplicados (releases del
     mismo procedimiento en distintas etapas de su ciclo de vida).
  4. Concentración: HHI (Herfindahl-Hirschman) de participación de mercado
     por proveedor a nivel nacional, y participación del monto adjudicado
     en los departamentos con mayor concentración geográfica (top-N share).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.hybrid_engine import _apply_structured_filters, load_config
from src.hybrid_indexer import HybridIndex, build_embedding_provider
from src.risk_analyzer import compute_risk_indicators


def evaluate_retrieval(config: dict, base_dir: Path, index: HybridIndex, provider) -> list[dict]:
    questions_path = base_dir / config["evaluation"]["questions_file"]
    with questions_path.open("r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    top_k = config["retrieval"]["top_k"]
    threshold = config["retrieval"]["similarity_threshold"]

    rows = []
    for q in questions:
        filters = {}
        if q.get("departamento_filtro"):
            filters["departamento"] = q["departamento_filtro"]
        if q.get("categoria_filtro"):
            filters["categoria"] = q["categoria_filtro"]

        candidates_df = _apply_structured_filters(index.metadata, filters)
        if candidates_df.empty:
            rows.append({
                "pregunta": q["pregunta"], "departamento_filtro": q.get("departamento_filtro", ""),
                "candidatos_tras_filtro": 0, "top1_similitud": 0.0, "supera_umbral": False,
            })
            continue

        candidate_vectors = index.vectors[candidates_df.index.to_numpy()]
        query_vector = provider.embed([q["pregunta"]])[0]
        sims = cosine_similarity(query_vector.reshape(1, -1), candidate_vectors)[0]
        top1 = float(np.sort(sims)[::-1][:top_k][0]) if len(sims) else 0.0

        rows.append({
            "pregunta": q["pregunta"], "departamento_filtro": q.get("departamento_filtro", ""),
            "candidatos_tras_filtro": len(candidates_df), "top1_similitud": round(top1, 4),
            "supera_umbral": top1 >= threshold,
        })
    return rows


def evaluate_cobertura(config: dict, contracts_df: pd.DataFrame, index: HybridIndex, retrieval_rows: list[dict]) -> dict:
    catalogo = set(config["departamentos"]["lista"])
    en_contratos = set(contracts_df["departamento"].dropna().unique()) if not contracts_df.empty else set()
    en_index = set(index.metadata["departamento"].dropna().unique()) if len(index) else set()
    con_resultados = {r["departamento_filtro"] for r in retrieval_rows if r["candidatos_tras_filtro"] > 0}

    return {
        "catalogo_total": len(catalogo),
        "en_contratos": sorted(en_contratos), "en_contratos_pct": len(en_contratos) / len(catalogo),
        "en_index": sorted(en_index), "en_index_pct": len(en_index) / len(catalogo),
        "sin_datos": sorted(catalogo - en_contratos),
        "con_resultados_retrieval": sorted(con_resultados), "con_resultados_retrieval_pct": len(con_resultados) / len(catalogo),
        "filtros_sin_resultados": sorted(catalogo & set(r["departamento_filtro"] for r in retrieval_rows) - con_resultados),
    }


def evaluate_risk_index_consistency(contracts_df: pd.DataFrame, risk_df_persisted: pd.DataFrame, riesgo_cfg: dict) -> dict:
    if contracts_df.empty:
        return {
            "departamentos_comparados": 0, "departamentos_identicos": 0, "diffs": [],
            "ocids_duplicados": [], "total_filas_contratos": 0,
        }

    recomputed = compute_risk_indicators(contracts_df, riesgo_cfg).set_index("departamento").sort_index()
    persisted = risk_df_persisted.set_index("departamento").sort_index()

    diffs = []
    affected_departments = set()
    for dept in recomputed.index:
        if dept not in persisted.index:
            diffs.append(f"{dept}: ausente en el parquet persistido")
            affected_departments.add(dept)
            continue
        for col in ("total_contratos", "contratos_pujador_unico", "monto_total", "tasa_pujador_unico"):
            a, b = recomputed.loc[dept, col], persisted.loc[dept, col]
            if not np.isclose(a, b, rtol=1e-6, atol=1e-6):
                diffs.append(f"{dept}.{col}: recalculado={a} vs. persistido={b}")
                affected_departments.add(dept)

    ocid_counts = contracts_df["ocid"].value_counts()
    duplicados = ocid_counts[ocid_counts > 1].index.tolist()

    return {
        "departamentos_comparados": len(recomputed),
        "departamentos_identicos": len(recomputed) - len(affected_departments),
        "diffs": diffs, "ocids_duplicados": duplicados,
        "total_filas_contratos": len(contracts_df),
    }


def _hhi(shares: pd.Series) -> float:
    """Herfindahl-Hirschman Index sobre participaciones de mercado (0-1 cada
    una); rango 0-10000 en la convención estándar (participación en %)."""
    pct = (shares / shares.sum()) * 100
    return float((pct ** 2).sum())


def evaluate_concentracion(contracts_df: pd.DataFrame, geo_metrics_df: pd.DataFrame) -> dict:
    if contracts_df.empty:
        return {"hhi_proveedores_nacional": None, "top5_departamentos_pct_monto": None, "interpretacion_hhi": ""}

    monto_por_proveedor = contracts_df.dropna(subset=["proveedor"]).groupby("proveedor")["monto"].sum()
    hhi = _hhi(monto_por_proveedor) if not monto_por_proveedor.empty else None

    if hhi is None:
        interpretacion = "sin datos de proveedor suficientes"
    elif hhi < 1500:
        interpretacion = "mercado no concentrado"
    elif hhi < 2500:
        interpretacion = "concentración moderada"
    else:
        interpretacion = "mercado altamente concentrado"

    monto_total = geo_metrics_df["monto_adjudicado_total"].sum()
    top5 = geo_metrics_df.nlargest(5, "monto_adjudicado_total")
    top5_pct = float(top5["monto_adjudicado_total"].sum() / monto_total) if monto_total else 0.0

    return {
        "hhi_proveedores_nacional": round(hhi, 1) if hhi is not None else None,
        "interpretacion_hhi": interpretacion,
        "top5_departamentos": top5[["departamento", "monto_adjudicado_total"]].to_dict("records"),
        "top5_departamentos_pct_monto": round(top5_pct, 4),
    }


def render_report(retrieval_rows: list[dict], cobertura: dict, riesgo: dict, concentracion: dict,
                   top_k: int, threshold: float) -> str:
    from datetime import datetime, timezone
    lines = [
        "# Reporte de Evaluación — Radar de Contrataciones (Tarea 2)",
        "", f"Generado: {datetime.now(timezone.utc).isoformat()}", "",
        "## 1. Cobertura de los 25 departamentos", "",
    ]
    lines.append(
        f"- En `contratos.parquet`: **{len(cobertura['en_contratos'])}/25** "
        f"({cobertura['en_contratos_pct']:.0%})."
    )
    lines.append(
        f"- En el índice híbrido (`hybrid_index.pkl`): **{len(cobertura['en_index'])}/25** "
        f"({cobertura['en_index_pct']:.0%})."
    )
    lines.append(
        f"- Con resultados reales de retrieval (filtro + similitud, no solo presencia en el catálogo): "
        f"**{len(cobertura['con_resultados_retrieval'])}/25** ({cobertura['con_resultados_retrieval_pct']:.0%})."
    )
    if cobertura["sin_datos"]:
        lines.append(f"- Departamentos sin ningún contrato en la muestra actual: {cobertura['sin_datos']}")
    if cobertura["filtros_sin_resultados"]:
        lines.append(f"- ⚠️ Filtros de departamento que no devolvieron candidatos pese a tener datos: {cobertura['filtros_sin_resultados']}")
    lines.append("")

    lines.append("## 2. Retrieval híbrido (filtros estructurados + embeddings)")
    lines.append("")
    n = len(retrieval_rows)
    con_candidatos = sum(1 for r in retrieval_rows if r["candidatos_tras_filtro"] > 0)
    supera_umbral = sum(1 for r in retrieval_rows if r["supera_umbral"])
    avg_top1 = sum(r["top1_similitud"] for r in retrieval_rows) / n if n else 0.0
    lines.append(f"- Preguntas evaluadas: {n} (una por departamento, con su categoría más frecuente en los datos reales).")
    lines.append(f"- Con al menos un candidato tras el filtro estructurado: {con_candidatos}/{n} ({con_candidatos/n:.0%}).")
    lines.append(f"- Con similitud top-1 ≥ umbral de abstención ({threshold}): {supera_umbral}/{n} ({supera_umbral/n:.0%}).")
    lines.append(f"- Similitud top-1 promedio: {avg_top1:.4f}.")
    lines.append("")
    lines.append("Detalle completo en `eval/resultados_radar.csv`.")
    lines.append("")

    lines.append("## 3. Precisión/consistencia del índice de riesgo (pujador único)")
    lines.append("")
    lines.append(
        "No existe una etiqueta externa de \"riesgo real\" contra la cual medir precisión en el sentido "
        "de clasificación supervisada (no hay un ground truth de qué contratos son efectivamente "
        "irregulares); lo que se puede y se verifica aquí es la **correctitud aritmética**: recalcular "
        "el indicador desde `contratos.parquet` con `risk_analyzer.compute_risk_indicators` y compararlo "
        "fila por fila contra `riesgo_departamento.parquet` ya persistido."
    )
    lines.append("")
    if riesgo["diffs"]:
        lines.append(f"⚠️ **{len(riesgo['diffs'])} discrepancias encontradas** entre el recálculo y el archivo persistido:")
        lines.extend(f"- {d}" for d in riesgo["diffs"][:20])
    else:
        lines.append(
            f"Sin discrepancias: las {riesgo['departamentos_comparados']} filas (una por departamento) de "
            "`riesgo_departamento.parquet` coinciden exactamente con el recálculo desde los contratos "
            "actuales — el indicador persistido está al día y es aritméticamente correcto."
        )
    lines.append("")
    if riesgo["ocids_duplicados"]:
        pct = len(riesgo["ocids_duplicados"]) / riesgo["total_filas_contratos"]
        lines.append(
            f"Hallazgo: {len(riesgo['ocids_duplicados'])} de {riesgo['total_filas_contratos']} filas "
            f"({pct:.2%}) en `contratos.parquet` comparten `ocid` con otra fila "
            f"(ejemplo: `{riesgo['ocids_duplicados'][0]}`). Esto es consistente con el comportamiento real de la "
            "API de OECE: un mismo procedimiento puede publicar releases separados por etapa del ciclo de vida "
            "(planning/tender, award, contract), cada uno como una fila independiente. Efecto actual mínimo, "
            "pero al aumentar `oece.max_pages` la tasa de duplicación podría crecer — si se necesita 'número de "
            "procedimientos únicos' en vez de 'número de releases', deduplicar por `ocid` antes de agregar."
        )
    lines.append("")

    lines.append("## 4. Índices de concentración")
    lines.append("")
    if concentracion["hhi_proveedores_nacional"] is not None:
        lines.append(
            f"- HHI de participación de mercado por proveedor (nacional, por monto adjudicado): "
            f"**{concentracion['hhi_proveedores_nacional']}** ({concentracion['interpretacion_hhi']}, "
            "escala estándar 0–10000: <1500 no concentrado, 1500–2500 moderado, >2500 alta concentración)."
        )
        lines.append(
            f"- Los 5 departamentos de mayor monto adjudicado concentran el "
            f"**{concentracion['top5_departamentos_pct_monto']:.1%}** del monto total adjudicado en la muestra."
        )
        lines.append("")
        lines.append("| Departamento | Monto adjudicado |")
        lines.append("|---|---|")
        for row in concentracion["top5_departamentos"]:
            lines.append(f"| {row['departamento']} | S/ {row['monto_adjudicado_total']:,.2f} |")
    else:
        lines.append("Sin datos suficientes para calcular concentración.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    config = load_config()

    index = HybridIndex.load(base_dir / config["paths"]["index_file"])
    provider = build_embedding_provider(config["embeddings"])
    contracts_df = pd.read_parquet(base_dir / config["paths"]["contracts_parquet"]) \
        if (base_dir / config["paths"]["contracts_parquet"]).exists() else pd.DataFrame()
    risk_df = pd.read_parquet(base_dir / config["paths"]["risk_parquet"]) \
        if (base_dir / config["paths"]["risk_parquet"]).exists() else pd.DataFrame()
    geo_metrics_df = pd.read_parquet(base_dir / config["paths"]["geo_metrics_parquet"]) \
        if (base_dir / config["paths"]["geo_metrics_parquet"]).exists() else pd.DataFrame()

    print("Evaluando retrieval híbrido...")
    retrieval_rows = evaluate_retrieval(config, base_dir, index, provider)

    results_path = base_dir / config["evaluation"]["results_file"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pregunta", "departamento_filtro", "candidatos_tras_filtro", "top1_similitud", "supera_umbral"])
        writer.writeheader()
        writer.writerows(retrieval_rows)
    print(f"Resultados de retrieval guardados en {results_path}")

    print("Evaluando cobertura de los 25 departamentos...")
    cobertura = evaluate_cobertura(config, contracts_df, index, retrieval_rows)

    print("Verificando consistencia del índice de riesgo...")
    riesgo_eval = evaluate_risk_index_consistency(contracts_df, risk_df, config["riesgo"])

    print("Calculando índices de concentración...")
    concentracion = evaluate_concentracion(contracts_df, geo_metrics_df)

    report_md = render_report(
        retrieval_rows, cobertura, riesgo_eval, concentracion,
        config["retrieval"]["top_k"], config["retrieval"]["similarity_threshold"],
    )
    report_path = base_dir / "docs" / "reporte_evaluacion_radar.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Reporte de evaluación guardado en {report_path}")


if __name__ == "__main__":
    main()
