# Reporte de Calidad de Extracción — Tarea 1 (RAG Normativo)

Generado: 2026-09-23T04:08:33.571690+00:00

## 1. Inspección y verificación de fuentes (source_check)

| Documento | Páginas | Caracteres totales | Prom. car./pág. | Mín. car./pág. | Máx. car./pág. | Páginas a 2 columnas | Páginas sin texto |
|---|---|---|---|---|---|---|---|
| LEY_32069.pdf | 63 | 214374 | 3403 | 1165 | 4190 | 0 | ninguna |
| DS_001_2026_EF.pdf | 16 | 110529 | 6908 | 6331 | 7902 | 16 | ninguna |

### LEY_32069.pdf
- Todas las páginas tienen capa de texto extraíble; no se requiere OCR.
- Layout de una sola columna en todas las páginas; no se requirió corrección de orden de lectura.
- Verificación de codificación (caracteres acentuados) a nivel de code point: 'REPÚBLICA
Le' (codepoints no-ASCII: U+00DA)

### DS_001_2026_EF.pdf
- Todas las páginas tienen capa de texto extraíble; no se requiere OCR.
- Layout a dos columnas detectado en 16 de 16 páginas (formato Diario Oficial El Peruano): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
- Orden de lectura corregido (evita intercalar columnas) en: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
- Verificación de codificación (caracteres acentuados) a nivel de code point: 'REPÚBLICA DE' (codepoints no-ASCII: U+00DA)

### Evidencia del problema de orden de lectura (antes / después)

Extracción ingenua (`page.extract_text()` de pdfplumber sin corrección de columnas), DS N.° 001-2026-EF, página 1 — el texto intercala ambas columnas y pierde coherencia:

```text
El Peruano / Jueves 8 de enero de 2026 NORMAS LEGALES 33
procedimientos ni requisitos, ni cambios sustantivos
ECONOMÍA Y FINANZAS que incrementen la carga administrativa, por lo que se
configura el supuesto contenido en el literal a) del numeral
Decreto Supremo que modifica el 50.2 del artículo 50 del Reglamento del Decreto Legislativo
Nº 1565, aprobado mediante el Decreto Supremo Nº 023-
Reglamento de la Ley N° 32069, Ley General
2025-PCM;
de Contrataciones Públicas, aprobado De conformidad con
```

Extracción corregida por columnas (implementada en `src/pdf_extractor.py::extract_page_text`), misma página — el texto es coherente de principio a fin:

```text
El Peruano / Jueves 8 de enero de 2026 NORMAS LEGALES 33
ECONOMÍA Y FINANZAS
Decreto Supremo que modifica el
Reglamento de la Ley N° 32069, Ley General
de Contrataciones Públicas, aprobado
mediante Decreto Supremo N° 009-2025-
EF
DECRETO SUPREMO
Nº 001-2026-EF
EL PRESIDENTE DE LA REPÚBLICA
CONSIDERANDO:
Que, el Decreto Legislativo Nº 1439, Decreto
Legislativo del Sistema Nacional de Abastecimiento,
desarrolla dicho Sistema con la finalidad de establecer
sus principios, definiciones, composición,
```

## 2. Extracción (page_number preservado)

Cada página se procesa de forma independiente y conserva `documento_id`, `documento_titulo`, `version`, `page_number` (número de página original del PDF, 1-indexado) y `es_dos_columnas`. Nunca se concatena el documento completo antes de fragmentar; `page_number` es el número real de página del archivo PDF de origen, verificado 1 a 1 contra `pdfplumber` (`len(pdf.pages)` = filas escritas por documento).

## 3. Limpieza de texto normativo

Reglas aplicadas en `src/text_cleaner.py` (no alteran numeración de artículos/numerales/incisos):
- Eliminación de la cabecera de maquetación de El Peruano (`El Peruano / <fecha> ... NORMAS LEGALES <n>`, en ambos órdenes según página par/impar).
- Eliminación del código de trámite editorial al pie de la última página (p. ej. `2474593-1`).
- Recomposición de palabras cortadas por guion de fin de línea.
- Colapso de espacios/saltos de línea redundantes.

- Cabeceras de El Peruano eliminadas en **16** páginas.
- Caracteres removidos en total por limpieza: **934**.
- Marcas de agua: no se detectó texto de marca de agua en la capa de texto de ningún PDF (sin caracteres rotados ni fuera de la paleta de color normal del cuerpo). Las imágenes presentes en el D.S. (logos/QR) no son procesadas por la extracción de texto, por lo que no contaminan `texto`; no fue necesaria una regla de limpieza para marcas de agua.

- Verificación de integridad estructural: en **todas** las páginas, el conjunto de tokens de estructura (`Artículo N`, numerales `N.N`, incisos `a)`…) detectados antes y después de la limpieza es **idéntico**. La limpieza no alteró la estructura de artículos, numerales ni incisos.

## 4. Conclusión

Los dos PDFs tienen capa de texto extraíble completa (no se requiere OCR). El D.S. N.° 001-2026-EF se publica a dos columnas en formato Diario Oficial El Peruano; se implementó detección automática de columnas por página (franja vacía central del cuerpo) para preservar el orden de lectura correcto antes de fragmentar. La Ley N.° 32069 (separata especial) es de una sola columna y no requirió corrección. La limpieza retira artefactos editoriales (cabecera de El Peruano, código de trámite) sin tocar la numeración normativa, verificado mediante comparación de tokens de estructura por página.
