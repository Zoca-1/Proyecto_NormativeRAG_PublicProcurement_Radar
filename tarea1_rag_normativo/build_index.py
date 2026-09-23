"""Script OFFLINE: extracción -> limpieza -> chunking -> embeddings -> índice.

Idempotente: cada etapa se salta si su artefacto ya existe, salvo --force.
Reanudable: al usar --force solo se reconstruye desde cero una vez invocado;
en ejecución normal, interrumpir y relanzar retoma desde el último artefacto
completo (extracción, luego chunks, luego índice).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.chunker import chunk_pages
from src.embeddings import build_embedding_provider
from src.pdf_extractor import extract_documents_to_jsonl, load_pages_jsonl
from src.rag_engine import load_config
from src.text_cleaner import clean_page_records
from src.vector_store import VectorStore

import json


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
        chunk_cfg = config["chunking"]
        chunks = chunk_pages(
            cleaned_pages, chunk_cfg["chunk_size_tokens"], chunk_cfg["chunk_overlap_tokens"], chunk_cfg["encoding_name"],
        )
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
        print("[4/5] Embeddings: ok (índice ya actualizado)")
    else:
        vectors = provider.embed([c["texto"] for c in pending_chunks])
        added = store.add(pending_chunks, vectors)
        store.save(index_path)
        print(f"[4/5] Embeddings: {added} vectores nuevos agregados al índice")

    print(f"[5/5] Índice final: {len(store)} chunks en {index_path}")


if __name__ == "__main__":
    main()
