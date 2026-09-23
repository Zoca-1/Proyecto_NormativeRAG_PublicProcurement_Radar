"""Fragmentación guiada por la estructura legal (Título > Artículo > Numeral > Inciso).

Cada fragmento nunca cruza una página: page_number es la etiqueta de
trazabilidad 1:1 hacia el PDF de origen que exige la arquitectura del
proyecto. Un artículo que continúa en la página siguiente genera fragmentos
separados, cada uno con su propio page_number real; el fragmento de
continuación hereda la etiqueta de articulo/titulo vigente al cierre de la
página anterior (se rastrea como estado secuencial mientras se procesan las
páginas de un mismo documento en orden). Si un fragmento estructural
(artículo, numeral o inciso) excede chunking.max_chunk_tokens, se subdivide
con una ventana de tokens de respaldo para no perder texto ni generar
fragmentos arbitrariamente grandes para el modelo de embeddings.
"""
from __future__ import annotations

import hashlib
import re

import tiktoken

_TITULO_HEADING = re.compile(r"(?m)^T[ÍI]TULO\s+(PRELIMINAR|[IVXLCDM]+)\b")
_ARTICULO_HEADING = re.compile(r"(?m)^Art[íi]culo\s+(\d+(?:-[A-Z])?)\.?\s*([^\n]*)")
_NUMERAL_HEADING = re.compile(r"(?m)^(\d{1,3}\.\d{1,2})\.\s")
_INCISO_HEADING = re.compile(r"(?m)^([a-z])\)\s")


def _chunk_id(documento_id: str, page_number: int, local_index: int) -> str:
    raw = f"{documento_id}:{page_number}:{local_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _split_segments(text: str, pattern: re.Pattern) -> list[tuple[re.Match | None, str]]:
    """Divide text en (match_o_None, segmento) en cada inicio de línea que
    cumpla pattern. El primer segmento, si el texto no arranca justo en una
    coincidencia, lleva match=None (continuación del contexto anterior)."""
    matches = list(pattern.finditer(text))
    if not matches:
        return [(None, text)]
    segments = []
    if matches[0].start() > 0:
        segments.append((None, text[:matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append((m, text[m.start():end]))
    return segments


def _token_len(text: str, encoding) -> int:
    return len(encoding.encode(text))


def _token_window_fallback(text: str, max_tokens: int, overlap_tokens: int, encoding) -> list[str]:
    """Última instancia si un fragmento estructural sigue siendo demasiado
    grande (p. ej. un inciso con una lista extensa): ventana de tokens con
    solape, igual que el chunking genérico previo a esta fase."""
    tokens = encoding.encode(text)
    if not tokens:
        return []
    step = max(max_tokens - overlap_tokens, 1)
    windows = []
    for start in range(0, len(tokens), step):
        window = tokens[start:start + max_tokens]
        if not window:
            continue
        windows.append(encoding.decode(window))
        if start + max_tokens >= len(tokens):
            break
    return windows


def _split_if_too_big(text: str, max_tokens: int, overlap_tokens: int, encoding) -> list[tuple[str | None, str]]:
    """Si text cabe en max_tokens, lo retorna tal cual (sin inciso). Si no,
    intenta dividir por incisos; si un inciso individual sigue siendo
    demasiado grande, cae a ventana de tokens."""
    if _token_len(text, encoding) <= max_tokens:
        return [(None, text)]

    pieces: list[tuple[str | None, str]] = []
    for match, segment in _split_segments(text, _INCISO_HEADING):
        inciso_label = f"{match.group(1)})" if match else None
        if _token_len(segment, encoding) <= max_tokens:
            pieces.append((inciso_label, segment))
        else:
            for window_text in _token_window_fallback(segment, max_tokens, overlap_tokens, encoding):
                pieces.append((inciso_label, window_text))
    return pieces


def chunk_page(page: dict, state: dict, chunking_cfg: dict, encoding) -> list[dict]:
    """Fragmenta una página según Título > Artículo > Numeral > Inciso.

    state es un dict {"titulo": str | None, "articulo": str | None} que se
    muta en el sitio y debe pasarse en orden de page_number ascendente para
    un mismo documento_id, de modo que un fragmento de continuación (una
    página que no abre con un encabezado nuevo) herede la etiqueta correcta.
    """
    max_tokens = chunking_cfg["max_chunk_tokens"]
    overlap_tokens = chunking_cfg["token_fallback_overlap_tokens"]

    chunks: list[dict] = []
    local_index = 0

    for titulo_match, titulo_segment in _split_segments(page["texto"], _TITULO_HEADING):
        if titulo_match:
            state["titulo"] = f"TÍTULO {titulo_match.group(1)}"

        for art_match, art_segment in _split_segments(titulo_segment, _ARTICULO_HEADING):
            if art_match:
                heading_text = art_match.group(2).strip()
                state["articulo"] = f"Artículo {art_match.group(1)}" + (f". {heading_text}" if heading_text else "")

            for num_match, num_segment in _split_segments(art_segment, _NUMERAL_HEADING):
                numeral_label = num_match.group(1) if num_match else None

                for inciso_label, final_text in _split_if_too_big(num_segment, max_tokens, overlap_tokens, encoding):
                    if not final_text.strip():
                        continue
                    chunks.append({
                        "chunk_id": _chunk_id(page["documento_id"], page["page_number"], local_index),
                        "documento_id": page["documento_id"],
                        "documento_titulo": page["documento_titulo"],
                        "version": page["version"],
                        "ley_ds": page["ley_ds"],
                        "page_number": page["page_number"],
                        "titulo": state["titulo"],
                        "articulo": state["articulo"],
                        "numeral": numeral_label,
                        "inciso": inciso_label,
                        "texto": final_text.strip(),
                    })
                    local_index += 1

    return chunks


def chunk_pages(pages: list[dict], chunking_cfg: dict) -> list[dict]:
    """Fragmenta una lista de páginas ya ordenadas por documento_id y
    page_number ascendente (ver load_pages_jsonl + sort en build_index.py)."""
    encoding = tiktoken.get_encoding(chunking_cfg["encoding_name"])
    all_chunks = []
    state_by_document: dict[str, dict] = {}
    for page in pages:
        state = state_by_document.setdefault(page["documento_id"], {"titulo": None, "articulo": None})
        all_chunks.extend(chunk_page(page, state, chunking_cfg, encoding))
    return all_chunks
