"""Limpieza de texto por página, sin fusionar páginas entre sí.

Además de la limpieza genérica (guiones de fin de línea, espacios/saltos
redundantes), elimina dos artefactos editoriales verificados en los PDFs
fuente durante source_check:
  - La cabecera de maquetación del Diario Oficial El Peruano
    ("El Peruano / <fecha> ... NORMAS LEGALES <n>", en ambos órdenes según
    página par/impar), presente únicamente en el D.S. N.° 001-2026-EF.
  - El código de trámite editorial que El Peruano imprime al pie de la
    última página (p. ej. "2474593-1").
No se eliminan marcas de agua porque source_check no detectó texto de marca
de agua en la capa de texto de ninguno de los dos PDFs (ver
docs/reporte_extraccion.md): las 206 imágenes de la página 1 del D.S. son
gráficas (logos/QR), no texto, y extract_text() no las procesa.
Ninguna de estas reglas toca la numeración de artículos/numerales/incisos.
"""
from __future__ import annotations

import re

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_PAGE_FOOTER_NUMBER = re.compile(r"\n\s*\d{1,4}\s*$")
_GAZETTE_MASTHEAD = re.compile(
    # Requiere "El Peruano" pegado a un "/" (como en la maqueta real) para no
    # confundirse con menciones del diario en citas o notas dentro del cuerpo
    # legal (p. ej. una nota de fe de erratas que dice 'edición ... de "El
    # Peruano"' sin la barra, la cual NO debe eliminarse).
    r"^(?:\d{0,4}\s*)?(?:El Peruano\s*/.*NORMAS\s+LEGALES\s*\d{0,4}|NORMAS\s+LEGALES.*?/\s*El Peruano)\s*$\n?",
    re.MULTILINE,
)
_EDITORIAL_TRACKING_CODE = re.compile(r"^\d{5,8}-\d{1,2}$\n?", re.MULTILINE)


def clean_page_text(texto: str) -> str:
    if not texto:
        return ""
    texto = _GAZETTE_MASTHEAD.sub("", texto)
    texto = _EDITORIAL_TRACKING_CODE.sub("", texto)
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
