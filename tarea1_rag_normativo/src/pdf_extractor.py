"""Extracción de texto por página, preservando documento/version/page_number.

Nunca concatena todo el PDF en un solo string: cada página es una unidad
independiente que conserva su metadata hasta el chunking.

El D.S. N.° 001-2026-EF se publica en el Diario Oficial El Peruano a dos
columnas. pdfplumber, por defecto, extrae texto por línea horizontal (top a
bottom, izquierda a derecha en cada línea), lo que intercala el contenido de
ambas columnas y rompe el orden de lectura. Para evitarlo, cada página se
inspecciona en busca de un "gutter" (franja vertical sin caracteres) dentro
del cuerpo de la página; si se encuentra, se extrae cabecera, columna
izquierda, columna derecha y pie de página por separado y se concatenan en
orden de lectura correcto.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pdfplumber


def _band_chars(page, top_min_ratio: float, top_max_ratio: float) -> list[dict]:
    h = page.height
    return [c for c in page.chars if h * top_min_ratio <= c["top"] < h * top_max_ratio]


def _find_central_gap(counts: list[int], bin_width: float, lo_idx: int, hi_idx: int,
                       max_density_per_bin: int, min_gap_width: float) -> float | None:
    runs: list[tuple[int, int]] = []
    start = None
    for i in range(lo_idx, hi_idx + 1):
        empty = counts[i] <= max_density_per_bin
        if empty and start is None:
            start = i
        if not empty and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, hi_idx))
    if not runs:
        return None

    best = max(runs, key=lambda r: r[1] - r[0])
    width = (best[1] - best[0] + 1) * bin_width
    if width < min_gap_width:
        return None
    return (best[0] + best[1] + 1) / 2 * bin_width


def detect_column_gap(page, column_cfg: dict) -> float | None:
    """Retorna la coordenada x del centro del gutter si la página es a dos
    columnas; None si es de una sola columna. La detección se hace solo sobre
    el cuerpo de la página (excluyendo cabecera y pie) para que una cabecera
    que cruza todo el ancho no oculte el gutter real del cuerpo.
    """
    body_chars = _band_chars(page, column_cfg["header_band_ratio"], 1 - column_cfg["footer_band_ratio"])
    if not body_chars:
        return None

    bins = column_cfg["histogram_bins"]
    bin_width = page.width / bins
    counts = [0] * bins
    for c in body_chars:
        idx = min(int(c["x0"] / bin_width), bins - 1)
        counts[idx] += 1

    lo_idx = int(column_cfg["gap_search_min_ratio"] * bins)
    hi_idx = int(column_cfg["gap_search_max_ratio"] * bins)
    min_gap_width = column_cfg["min_gap_width_ratio"] * page.width
    return _find_central_gap(counts, bin_width, lo_idx, hi_idx, column_cfg["max_density_per_bin"], min_gap_width)


def extract_page_text(page, column_cfg: dict) -> tuple[str, bool]:
    """Extrae el texto de una página, corrigiendo el orden de lectura si es a
    dos columnas. Retorna (texto, es_dos_columnas).

    Solo se separa la cabecera del cuerpo (header_band_ratio): una cabecera a
    todo lo ancho, cropeada junto con las columnas, queda partida a la mitad
    de una palabra por el corte del gutter (se comprobó extrayendo el D.S.
    N.° 001-2026-EF). No se recorta un pie de página aparte: estos boletines
    no tienen un pie fijo por página, y el contenido real (p. ej. el bloque
    de firma y el código editorial de la última página) llega hasta el borde
    inferior; forzar un footer_band_ratio > 0 en la extracción reproduce el
    mismo intercalado de columnas que se busca evitar cuando dos líneas de
    columnas distintas caen a una altura casi idéntica cerca del margen
    inferior. footer_band_ratio se usa únicamente para excluir el margen
    inferior del cálculo del gutter en detect_column_gap.
    """
    gap_x = detect_column_gap(page, column_cfg)
    if gap_x is None:
        return page.extract_text() or "", False

    w, h = page.width, page.height
    header_end = h * column_cfg["header_band_ratio"]

    header = page.crop((0, 0, w, header_end)).extract_text() or ""
    left = page.crop((0, header_end, gap_x, h)).extract_text() or ""
    right = page.crop((gap_x, header_end, w, h)).extract_text() or ""

    text = "\n".join(part for part in (header, left, right) if part.strip())
    return text, True


def extract_pages(pdf_path: Path, documento_id: str, titulo: str, version: str, ley_ds: str, column_cfg: dict) -> Iterator[dict]:
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            texto, es_dos_columnas = extract_page_text(page, column_cfg)
            yield {
                "documento_id": documento_id,
                "documento_titulo": titulo,
                "version": version,
                "ley_ds": ley_ds,
                "page_number": page_number,
                "es_dos_columnas": es_dos_columnas,
                "texto": texto,
            }


def extract_documents_to_jsonl(documents: list[dict], raw_dir: Path, output_path: Path, column_cfg: dict) -> int:
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
            for page_record in extract_pages(pdf_path, doc["id"], doc["titulo"], doc["version"], doc["ley_ds"], column_cfg):
                f.write(json.dumps(page_record, ensure_ascii=False) + "\n")
                total_pages += 1
    return total_pages


def load_pages_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
