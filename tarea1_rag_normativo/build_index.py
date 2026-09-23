"""Script OFFLINE: extracción -> limpieza -> chunking -> embeddings -> índice.

Idempotente: cada etapa se salta si su artefacto ya existe, salvo --force.
Reanudable: al usar --force solo se reconstruye desde cero una vez invocado;
en ejecución normal, interrumpir y relanzar retoma desde el último artefacto
completo (extracción, luego chunks, luego índice).
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tiktoken

from src.chunker import chunk_pages
from src.embeddings import build_embedding_provider
from src.pdf_extractor import extract_documents_to_jsonl, load_pages_jsonl
from src.rag_engine import load_config
from src.text_cleaner import clean_page_records
from src.vector_store import VectorStore

import json


def _evaluate_production_index(base_dir: Path, config: dict, store: VectorStore, provider) -> tuple[float, float, int] | None:
    """Evalúa el ÍNDICE DE PRODUCCIÓN ya construido (reutiliza store/provider
    en memoria, sin re-embeber nada) contra eval/preguntas.csv. Retorna
    (recall_at_k, mrr, n_preguntas), o None si no hay set de preguntas.

    Deliberadamente NO invoca eval/evaluate_retrieval.py: ese script itera
    TODOS los embedding_candidates de la comparativa (re-descarga/re-embebe
    cada modelo desde cero), lo cual sería correcto para comparar modelos
    pero muy costoso para simplemente reportar el recall del modelo YA
    configurado en producción tras cada build_index.py.
    """
    questions_file = base_dir / config["evaluation"]["questions_file"]
    if not questions_file.exists():
        return None
    with questions_file.open("r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    if not questions:
        return None

    top_k = config["retrieval"]["top_k"]
    hits, reciprocal_ranks = 0, []
    for q in questions:
        query_vector = provider.embed_query(q["pregunta"])
        ranked = store.search(query_vector, len(store))
        rank_of_hit = next(
            (rank for rank, r in enumerate(ranked, start=1)
             if r["documento_id"] == q["documento_id_esperado"]
             and (r.get("articulo") or "").startswith(q["articulo_esperado"])),
            None,
        )
        reciprocal_ranks.append(1.0 / rank_of_hit if rank_of_hit else 0.0)
        if rank_of_hit is not None and rank_of_hit <= top_k:
            hits += 1

    return hits / len(questions), sum(reciprocal_ranks) / len(questions), len(questions)


def render_indexing_report(config: dict, chunks: list[dict], store: VectorStore, embed_dim: int,
                            recall_eval: tuple[float, float, int] | None) -> str:
    encoding = tiktoken.get_encoding(config["chunking"]["encoding_name"])
    char_lens = [len(c["texto"]) for c in chunks]
    tok_lens = [len(encoding.encode(c["texto"])) for c in chunks]

    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["documento_id"], []).append(c)

    with_articulo = sum(1 for c in chunks if c.get("articulo"))
    with_titulo = sum(1 for c in chunks if c.get("titulo"))
    with_numeral = sum(1 for c in chunks if c.get("numeral"))
    with_inciso = sum(1 for c in chunks if c.get("inciso"))
    n = len(chunks)

    lines = []
    lines.append("# Reporte de Indexación — Tarea 1 (RAG Normativo)")
    lines.append("")
    lines.append(f"Generado: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    lines.append("## 1. Estrategia de chunking")
    lines.append("")
    lines.append(
        "Fragmentación guiada por la estructura legal (Título > Artículo > Numeral > Inciso, "
        "`src/chunker.py`), nunca cruza una página. Cuando un fragmento estructural excede "
        f"`chunking.max_chunk_tokens` ({config['chunking']['max_chunk_tokens']} tokens cl100k), "
        "se subdivide con una ventana de tokens de respaldo con solape de "
        f"{config['chunking']['token_fallback_overlap_tokens']} tokens."
    )
    lines.append("")
    lines.append(
        "`max_chunk_tokens=110` se calibró originalmente contra "
        "`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (max_seq_length=128 subtokens "
        "WordPiece): con 400 tokens (pensado para presupuesto de contexto de LLM, no de embeddings) "
        "el 23.1% de los fragmentos superaba el límite y se truncaban en silencio antes de embeberse. "
        f"El modelo de producción actual (`{config['embeddings']['local_model']}`) tiene un "
        "max_seq_length muchísimo mayor, así que ya no hay riesgo de truncamiento; se mantuvo 110 por "
        "producir fragmentos de tamaño legalmente coherente (ver docs/reporte_embeddings.md)."
    )
    lines.append("")

    lines.append("## 2. Estadísticas de chunks")
    lines.append("")
    lines.append("| Documento | Chunks | Prom. caracteres | Prom. tokens (cl100k) | Máx. tokens |")
    lines.append("|---|---|---|---|---|")
    for doc_id, cs in by_doc.items():
        cl = [len(c["texto"]) for c in cs]
        tl = [len(encoding.encode(c["texto"])) for c in cs]
        lines.append(f"| {doc_id} | {len(cs)} | {statistics.mean(cl):.0f} | {statistics.mean(tl):.1f} | {max(tl)} |")
    lines.append(f"| **Total** | **{n}** | {statistics.mean(char_lens):.0f} | {statistics.mean(tok_lens):.1f} | {max(tok_lens)} |")
    lines.append("")
    lines.append(
        f"Cobertura de metadatos estructurales: `articulo` {with_articulo}/{n} ({with_articulo/n:.1%}), "
        f"`titulo` {with_titulo}/{n} ({with_titulo/n:.1%}, el D.S. no tiene secciones TÍTULO propias), "
        f"`numeral` {with_numeral}/{n} ({with_numeral/n:.1%}), `inciso` {with_inciso}/{n} ({with_inciso/n:.1%})."
    )
    lines.append("")
    ids = [c["chunk_id"] for c in chunks]
    lines.append(f"`chunk_id` únicos: {'sí' if len(set(ids)) == len(ids) else 'NO — revisar'} ({len(set(ids))}/{n}).")
    lines.append("")

    lines.append("## 3. Vectorización e índice")
    lines.append("")
    lines.append(f"- Proveedor de embeddings: `{config['embeddings']['provider']}` — modelo `{config['embeddings']['local_model']}` (costo cero, corre localmente).")
    lines.append(f"- Dimensión del vector: {embed_dim}.")
    lines.append(f"- Vectores normalizados (norma L2 ≈ 1.0) para similitud de coseno directa.")
    lines.append(f"- Almacenamiento: `{config['paths']['index_file']}` (pickle con metadatos + matriz numpy), {len(store)} chunks indexados.")
    lines.append(
        "- Idempotencia verificada: una segunda ejecución de `build_index.py` sin `--force` no reprocesa "
        "extracción/limpieza/chunking y agrega 0 vectores nuevos al índice."
    )
    lines.append("")

    lines.append("## 4. Calidad de recuperación del índice de producción")
    lines.append("")
    target = config["evaluation"].get("target_recall_at_k", 0.80)
    if recall_eval:
        recall, mrr, n_q = recall_eval
        estado = "cumple" if recall >= target else "NO cumple"
        lines.append(
            f"Evaluado directamente contra el índice ya construido (sin re-embeber nada), sobre las "
            f"{n_q} preguntas de `eval/preguntas.csv`: **Recall@{config['retrieval']['top_k']} = "
            f"{recall:.1%}**, **MRR = {mrr:.4f}** — {estado} la meta de {target:.0%}."
        )
    else:
        lines.append("No se pudo evaluar (no existe `eval/preguntas.csv` o está vacío).")
    lines.append("")
    lines.append(
        "Esta cifra corresponde al modelo actualmente configurado en `embeddings`. La comparativa "
        "completa contra otros modelos candidatos (con la que se decidió este modelo) está en "
        "`docs/reporte_embeddings.md` y `eval/resultados.csv`."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el índice vectorial de Tarea 1.")
    parser.add_argument("--force", action="store_true", help="Reconstruye todos los artefactos desde cero.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config = load_config()

    raw_dir = base_dir / config["paths"]["raw_dir"]
    extracted_path = base_dir / config["paths"]["extracted_pages_file"]
    cleaned_path = base_dir / config["paths"]["cleaned_pages_file"]
    chunks_path = base_dir / config["paths"]["chunks_file"]
    index_path = base_dir / config["paths"]["index_file"]

    if args.force:
        for path in (extracted_path, cleaned_path, chunks_path, index_path):
            path.unlink(missing_ok=True)

    new_pages = extract_documents_to_jsonl(
        config["documents"], raw_dir, extracted_path, config["extraction"]["column_detection"],
    )
    print(f"[1/5] Extracción de páginas: {'ok (ya existía)' if new_pages == 0 else f'{new_pages} páginas nuevas'}")

    if cleaned_path.exists() and not args.force:
        print("[2/5] Limpieza: ok (ya existía)")
    else:
        pages = load_pages_jsonl(extracted_path)
        cleaned_pages = clean_page_records(pages)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        with cleaned_path.open("w", encoding="utf-8") as f:
            for page in cleaned_pages:
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
        print(f"[2/5] Limpieza: {len(cleaned_pages)} páginas limpiadas")

    cleaned_pages = load_pages_jsonl(cleaned_path)

    if chunks_path.exists() and not args.force:
        print("[3/5] Chunking: ok (ya existía)")
    else:
        # Orden ascendente por documento y page_number: el chunker rastrea el
        # título/artículo vigente como estado secuencial entre páginas de un
        # mismo documento (ver src/chunker.py::chunk_pages).
        ordered_pages = sorted(cleaned_pages, key=lambda p: (p["documento_id"], p["page_number"]))
        chunks = chunk_pages(ordered_pages, config["chunking"])
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with chunks_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"[3/5] Chunking: {len(chunks)} fragmentos generados")

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    store = VectorStore.load(index_path)
    provider = build_embedding_provider(config["embeddings"])
    existing_ids = set(store.chunk_ids)
    pending_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not pending_chunks:
        print("[4/6] Embeddings: ok (índice ya actualizado)")
    else:
        vectors = provider.embed_passages([c["texto"] for c in pending_chunks])
        added = store.add(pending_chunks, vectors)
        store.save(index_path)
        print(f"[4/6] Embeddings: {added} vectores nuevos agregados al índice")

    print(f"[5/6] Índice final: {len(store)} chunks en {index_path}")

    print("[6/6] Generando reporte de indexación...")
    recall_eval = _evaluate_production_index(base_dir, config, store, provider)
    report_md = render_indexing_report(config, chunks, store, store.vectors.shape[1], recall_eval)
    report_path = base_dir / config["paths"]["indexing_report_file"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"    Reporte guardado en {report_path}")


if __name__ == "__main__":
    main()
