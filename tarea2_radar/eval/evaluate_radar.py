"""Evaluación de retrieval híbrido, costo-cero (no llama al LLM).

Para cada pregunta aplica los filtros estructurados y mide la similitud
máxima obtenida y cuántos candidatos sobreviven el filtro, sin gastar
tokens de LLM.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.hybrid_engine import _apply_structured_filters, load_config
from src.hybrid_indexer import HybridIndex, build_embedding_provider


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    config = load_config()

    with (base_dir / config["evaluation"]["questions_file"]).open("r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    index = HybridIndex.load(base_dir / config["paths"]["index_file"])
    provider = build_embedding_provider(config["embeddings"])
    top_k = config["retrieval"]["top_k"]

    rows = []
    for q in questions:
        filters = {}
        if q.get("departamento_filtro"):
            filters["departamento"] = q["departamento_filtro"]
        if q.get("categoria_filtro"):
            filters["categoria"] = q["categoria_filtro"]

        candidates_df = _apply_structured_filters(index.metadata, filters)
        if candidates_df.empty:
            rows.append({"pregunta": q["pregunta"], "candidatos_tras_filtro": 0, "top1_similitud": 0.0})
            continue

        candidate_vectors = index.vectors[candidates_df.index.to_numpy()]
        query_vector = provider.embed([q["pregunta"]])[0]
        sims = cosine_similarity(query_vector.reshape(1, -1), candidate_vectors)[0]
        top1 = float(np.sort(sims)[::-1][:top_k][0]) if len(sims) else 0.0

        rows.append({
            "pregunta": q["pregunta"],
            "candidatos_tras_filtro": len(candidates_df),
            "top1_similitud": round(top1, 4),
        })

    results_path = base_dir / config["evaluation"]["results_file"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pregunta", "candidatos_tras_filtro", "top1_similitud"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Resultados guardados en {results_path}")


if __name__ == "__main__":
    main()
