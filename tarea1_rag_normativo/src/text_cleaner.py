"""Limpieza de texto por página, sin fusionar páginas entre sí."""
from __future__ import annotations

import re

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_PAGE_FOOTER_NUMBER = re.compile(r"\n\s*\d{1,4}\s*$")


def clean_page_text(texto: str) -> str:
    if not texto:
        return ""
    texto = _HYPHEN_LINEBREAK.sub(r"\1\2", texto)
    texto = _PAGE_FOOTER_NUMBER.sub("", texto)
    texto = _MULTI_SPACE.sub(" ", texto)
    texto = _MULTI_NEWLINE.sub("\n\n", texto)
    return texto.strip()


def clean_page_records(pages: list[dict]) -> list[dict]:
    cleaned = []
    for page in pages:
        record = dict(page)
        record["texto"] = clean_page_text(page["texto"])
        cleaned.append(record)
    return cleaned
