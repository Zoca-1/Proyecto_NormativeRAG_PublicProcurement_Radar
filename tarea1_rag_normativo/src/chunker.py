"""Fragmentación por tokens, dentro de cada página (nunca cruza páginas).

Cada chunk conserva documento_id, documento_titulo, version y page_number,
más un chunk_id determinístico (hash) para permitir indexado idempotente.
"""
from __future__ import annotations

import hashlib

import tiktoken


def _chunk_id(documento_id: str, page_number: int, chunk_index: int) -> str:
    raw = f"{documento_id}:{page_number}:{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def chunk_page(page: dict, chunk_size_tokens: int, chunk_overlap_tokens: int, encoding_name: str) -> list[dict]:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(page["texto"])
    if not tokens:
        return []

    step = max(chunk_size_tokens - chunk_overlap_tokens, 1)
    chunks = []
    chunk_index = 0
    for start in range(0, len(tokens), step):
        window = tokens[start:start + chunk_size_tokens]
        if not window:
            continue
        chunks.append({
            "chunk_id": _chunk_id(page["documento_id"], page["page_number"], chunk_index),
            "documento_id": page["documento_id"],
            "documento_titulo": page["documento_titulo"],
            "version": page["version"],
            "page_number": page["page_number"],
            "texto": encoding.decode(window),
        })
        chunk_index += 1
        if start + chunk_size_tokens >= len(tokens):
            break
    return chunks


def chunk_pages(pages: list[dict], chunk_size_tokens: int, chunk_overlap_tokens: int, encoding_name: str) -> list[dict]:
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_page(page, chunk_size_tokens, chunk_overlap_tokens, encoding_name))
    return all_chunks
