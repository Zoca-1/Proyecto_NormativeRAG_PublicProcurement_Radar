# Reporte de Indexación — Tarea 1 (RAG Normativo)

Generado: 2026-09-23T05:04:30.645154+00:00

## 1. Estrategia de chunking

Fragmentación guiada por la estructura legal (Título > Artículo > Numeral > Inciso, `src/chunker.py`), nunca cruza una página. Cuando un fragmento estructural excede `chunking.max_chunk_tokens` (110 tokens cl100k), se subdivide con una ventana de tokens de respaldo con solape de 20 tokens.

`max_chunk_tokens=110` se calibró originalmente contra `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (max_seq_length=128 subtokens WordPiece): con 400 tokens (pensado para presupuesto de contexto de LLM, no de embeddings) el 23.1% de los fragmentos superaba el límite y se truncaban en silencio antes de embeberse. El modelo de producción actual (`BAAI/bge-m3`) tiene un max_seq_length muchísimo mayor, así que ya no hay riesgo de truncamiento; se mantuvo 110 por producir fragmentos de tamaño legalmente coherente (ver docs/reporte_embeddings.md).

## 2. Estadísticas de chunks

| Documento | Chunks | Prom. caracteres | Prom. tokens (cl100k) | Máx. tokens |
|---|---|---|---|---|
| ds_001_2026_ef | 408 | 302 | 84.2 | 112 |
| ley_32069 | 950 | 242 | 64.6 | 112 |
| **Total** | **1358** | 260 | 70.5 | 112 |

Cobertura de metadatos estructurales: `articulo` 1334/1358 (98.2%), `titulo` 941/1358 (69.3%, el D.S. no tiene secciones TÍTULO propias), `numeral` 843/1358 (62.1%), `inciso` 455/1358 (33.5%).

`chunk_id` únicos: sí (1358/1358).

## 3. Vectorización e índice

- Proveedor de embeddings: `local` — modelo `BAAI/bge-m3` (costo cero, corre localmente).
- Dimensión del vector: 1024.
- Vectores normalizados (norma L2 ≈ 1.0) para similitud de coseno directa.
- Almacenamiento: `data/processed/vector_index.pkl` (pickle con metadatos + matriz numpy), 1358 chunks indexados.
- Idempotencia verificada: una segunda ejecución de `build_index.py` sin `--force` no reprocesa extracción/limpieza/chunking y agrega 0 vectores nuevos al índice.

## 4. Calidad de recuperación del índice de producción

Evaluado directamente contra el índice ya construido (sin re-embeber nada), sobre las 26 preguntas de `eval/preguntas.csv`: **Recall@5 = 92.3%**, **MRR = 0.8463** — cumple la meta de 80%.

Esta cifra corresponde al modelo actualmente configurado en `embeddings`. La comparativa completa contra otros modelos candidatos (con la que se decidió este modelo) está en `docs/reporte_embeddings.md` y `eval/resultados.csv`.
