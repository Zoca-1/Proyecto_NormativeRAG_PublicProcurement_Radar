"""Comparativa de proveedores de embeddings sobre retrieval, costo-cero para
los modelos locales (no llama al LLM en ningún caso).

Para cada candidato en config.yaml -> evaluation.embedding_candidates:
  1. Construye el proveedor de embeddings (local o vía API).
  2. Embebe TODO el corpus de chunks.jsonl como pasajes (embed_passages, con
     el passage_prefix del candidato) y arma un índice en memoria — no toca
     el índice de producción (vector_index.pkl).
  3. Embebe cada pregunta de eval/preguntas.csv como consulta (embed_query,
     con el query_prefix del candidato) y busca top_k.
  4. Un acierto es que ALGÚN resultado en el top_k tenga el mismo
     documento_id y un articulo que empiece con el articulo_esperado (más
     robusto que comparar page_number, ya que el chunking por estructura
     legal puede partir un mismo artículo en varias páginas/fragmentos).
  5. Calcula Recall@k y MRR (Mean Reciprocal Rank, considerando todo el
     corpus, no solo el top_k) por candidato, y los registra en
     eval/resultados.csv.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_pages  # noqa: F401  (chunks.jsonl ya viene generado por build_index.py)
from src.embeddings import build_embedding_provider
from src.rag_engine import load_config
from src.vector_store import VectorStore


def _is_hit(chunk_meta: dict, documento_id_esperado: str, articulo_esperado: str) -> bool:
    if chunk_meta["documento_id"] != documento_id_esperado:
        return False
    articulo = chunk_meta.get("articulo") or ""
    return articulo.startswith(articulo_esperado)


def load_corpus_chunks(chunks_path: Path) -> list[dict]:
    import json
    with chunks_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_candidate_store(candidate: dict, chunks: list[dict]) -> tuple[VectorStore, object, float]:
    embeddings_cfg = {
        "provider": candidate["provider"],
        "local_model": candidate.get("model"),
        "openai_model": candidate.get("model"),
        "batch_size": candidate.get("batch_size", 16),
        "query_prefix": candidate.get("query_prefix", ""),
        "passage_prefix": candidate.get("passage_prefix", ""),
    }
    provider = build_embedding_provider(embeddings_cfg)

    start = time.monotonic()
    vectors = provider.embed_passages([c["texto"] for c in chunks])
    elapsed = time.monotonic() - start

    store = VectorStore()
    store.add(chunks, vectors)
    return store, provider, elapsed


def evaluate_candidate(candidate: dict, store: VectorStore, provider, questions: list[dict], top_k: int) -> list[dict]:
    rows = []
    for q in questions:
        query_vector = provider.embed_query(q["pregunta"])
        ranked = store.search(query_vector, len(store))

        rank_of_hit = None
        for rank, r in enumerate(ranked, start=1):
            if _is_hit(r, q["documento_id_esperado"], q["articulo_esperado"]):
                rank_of_hit = rank
                break

        top_k_results = ranked[:top_k]
        acierto = rank_of_hit is not None and rank_of_hit <= top_k
        reciprocal_rank = (1.0 / rank_of_hit) if rank_of_hit else 0.0
        top1_similitud = top_k_results[0]["similitud"] if top_k_results else 0.0

        rows.append({
            "proveedor": candidate["name"],
            "pregunta": q["pregunta"],
            "acierto_recall_at_k": acierto,
            "rank_del_correcto": rank_of_hit or "",
            "reciprocal_rank": round(reciprocal_rank, 4),
            "top1_similitud": round(top1_similitud, 4),
        })
    return rows


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    config = load_config()

    with (base_dir / config["evaluation"]["questions_file"]).open("r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    chunks_path = base_dir / config["paths"]["chunks_file"]
    if not chunks_path.exists():
        raise FileNotFoundError(f"No existe {chunks_path}. Ejecuta build_index.py (al menos hasta chunking) primero.")
    chunks = load_corpus_chunks(chunks_path)

    top_k = config["retrieval"]["top_k"]
    summary = []

    results_path = base_dir / config["evaluation"]["results_file"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["proveedor", "pregunta", "acierto_recall_at_k", "rank_del_correcto", "reciprocal_rank", "top1_similitud"]
    with results_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for candidate in config["evaluation"]["embedding_candidates"]:
        print(f"Evaluando candidato '{candidate['name']}' ({candidate['provider']}: {candidate.get('model')})...", flush=True)
        try:
            store, provider, embed_seconds = build_candidate_store(candidate, chunks)
        except Exception as exc:  # noqa: BLE001 - se reporta y se continúa con el siguiente candidato
            print(f"    ERROR construyendo el candidato: {exc}", flush=True)
            summary.append({"proveedor": candidate["name"], "recall_at_k": None, "mrr": None, "error": str(exc)})
            continue

        rows = evaluate_candidate(candidate, store, provider, questions, top_k)

        # Escritura incremental: si un candidato pesado (p. ej. bge-m3) tarda
        # mucho, los resultados de los candidatos previos ya quedan en disco.
        with results_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(rows)

        recall = sum(r["acierto_recall_at_k"] for r in rows) / len(rows) if rows else 0.0
        mrr = sum(r["reciprocal_rank"] for r in rows) / len(rows) if rows else 0.0
        summary.append({
            "proveedor": candidate["name"], "recall_at_k": recall, "mrr": mrr,
            "embed_seconds": round(embed_seconds, 1), "error": None,
        })
        print(f"    Recall@{top_k} = {recall:.1%} | MRR = {mrr:.4f} | tiempo embeddings corpus = {embed_seconds:.1f}s", flush=True)

    print(f"\nResultados detallados guardados en {results_path}")

    print("\nResumen:")
    target = config["evaluation"].get("target_recall_at_k", 0.80)
    for s in summary:
        if s["error"]:
            print(f"  {s['proveedor']}: ERROR — {s['error']}")
        else:
            marca = "OK" if s["recall_at_k"] >= target else "por debajo de la meta"
            print(f"  {s['proveedor']}: recall@{top_k}={s['recall_at_k']:.1%} MRR={s['mrr']:.4f} [{marca}] (meta {target:.0%})")


if __name__ == "__main__":
    main()
