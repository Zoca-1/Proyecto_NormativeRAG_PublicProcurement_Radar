"""Evaluación de retrieval, costo-cero (no llama al LLM).

Mide recall@k y similitud promedio del top-1 para cada proveedor de
embeddings listado en config.yaml -> evaluation.embedding_providers_to_compare,
permitiendo comparar, por ejemplo, embeddings locales vs. de pago.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import build_embedding_provider
from src.rag_engine import load_config
from src.vector_store import VectorStore


def evaluate_provider(provider_name: str, config: dict, base_dir: Path, questions: list[dict]) -> list[dict]:
    embeddings_cfg = dict(config["embeddings"])
    embeddings_cfg["provider"] = provider_name
    provider = build_embedding_provider(embeddings_cfg)
    store = VectorStore.load(base_dir / config["paths"]["index_file"])

    top_k = config["retrieval"]["top_k"]
    rows = []
    for q in questions:
        vector = provider.embed([q["pregunta"]])[0]
        results = store.search(vector, top_k)
        paginas_recuperadas = {(r["documento_id"], r["pagina"]) for r in results}
        esperado = (q["documento_id_esperado"], int(q["pagina_esperada"]))
        acierto = esperado in paginas_recuperadas
        top1_similitud = results[0]["similitud"] if results else 0.0
        rows.append({
            "proveedor": provider_name,
            "pregunta": q["pregunta"],
            "acierto_recall_at_k": acierto,
            "top1_similitud": round(top1_similitud, 4),
        })
    return rows


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    config = load_config()

    with (base_dir / config["evaluation"]["questions_file"]).open("r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    all_rows = []
    for provider_name in config["evaluation"]["embedding_providers_to_compare"]:
        rows = evaluate_provider(provider_name, config, base_dir, questions)
        all_rows.extend(rows)
        recall = sum(r["acierto_recall_at_k"] for r in rows) / len(rows) if rows else 0.0
        print(f"Proveedor '{provider_name}': recall@k = {recall:.2%}")

    results_path = base_dir / config["evaluation"]["results_file"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["proveedor", "pregunta", "acierto_recall_at_k", "top1_similitud"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Resultados guardados en {results_path}")


if __name__ == "__main__":
    main()
