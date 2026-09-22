"""Extracción de texto por página, preservando documento/version/pagina.

Nunca concatena todo el PDF en un solo string: cada página es una unidad
independiente que conserva su metadata hasta el chunking.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pdfplumber


def extract_pages(pdf_path: Path, documento_id: str, titulo: str, version: str) -> Iterator[dict]:
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            texto = page.extract_text() or ""
            yield {
                "documento_id": documento_id,
                "documento_titulo": titulo,
                "version": version,
                "pagina": page_number,
                "texto": texto,
            }


def extract_documents_to_jsonl(documents: list[dict], raw_dir: Path, output_path: Path) -> int:
    """Idempotente: si output_path ya existe, no reprocesa y retorna 0 páginas nuevas."""
    if output_path.exists():
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_pages = 0
    with output_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            pdf_path = raw_dir / doc["file"]
            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"No se encontró '{pdf_path}'. Coloca el PDF declarado en config.yaml -> documents."
                )
            for page_record in extract_pages(pdf_path, doc["id"], doc["titulo"], doc["version"]):
                f.write(json.dumps(page_record, ensure_ascii=False) + "\n")
                total_pages += 1
    return total_pages


def load_pages_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
