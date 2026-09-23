# Reporte de Indexación — Tarea 1 (RAG Normativo)

Generado: 2026-09-23T04:20:09.661716+00:00

## 1. Estrategia de chunking

Fragmentación guiada por la estructura legal (Título > Artículo > Numeral > Inciso, `src/chunker.py`), nunca cruza una página. Cuando un fragmento estructural excede `chunking.max_chunk_tokens` (110 tokens cl100k), se subdivide con una ventana de tokens de respaldo con solape de 20 tokens.

`max_chunk_tokens` se calibró contra el `max_seq_length` real del modelo de embeddings configurado (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` = 128 subtokens WordPiece): con un límite de 400 tokens (pensado originalmente para presupuesto de contexto de LLM, no de embeddings) el 23.1% de los fragmentos (181/784) superaba los 128 subtokens y el modelo los truncaba en silencio antes de generar el embedding. Con 110 tokens cl100k, el fragmento WordPiece más largo observado es 102 (0% de truncamiento).

## 2. Estadísticas de chunks

| Documento | Chunks | Prom. caracteres | Prom. tokens (cl100k) | Máx. tokens |
|---|---|---|---|---|
| ds_001_2026_ef | 408 | 302 | 84.2 | 112 |
| ley_32069 | 950 | 242 | 64.6 | 112 |
| **Total** | **1358** | 260 | 70.5 | 112 |

Cobertura de metadatos estructurales: `articulo` 1334/1358 (98.2%), `titulo` 941/1358 (69.3%, el D.S. no tiene secciones TÍTULO propias), `numeral` 843/1358 (62.1%), `inciso` 455/1358 (33.5%).

`chunk_id` únicos: sí (1358/1358).

## 3. Vectorización e índice

- Proveedor de embeddings: `local` — modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (costo cero, corre localmente).
- Dimensión del vector: 384.
- Vectores normalizados (norma L2 ≈ 1.0) para similitud de coseno directa.
- Almacenamiento: `data/processed/vector_index.pkl` (pickle con metadatos + matriz numpy), 1358 chunks indexados.
- Idempotencia verificada: una segunda ejecución de `build_index.py` sin `--force` no reprocesa extracción/limpieza/chunking y agrega 0 vectores nuevos al índice.

## 4. Nota de calidad de recuperación (hallazgo, no bloqueante)

`eval/evaluate_retrieval.py` (costo cero, sin LLM) sobre las 5 preguntas de `eval/preguntas.csv` da **recall@5 = 0%** con el modelo local actual.

Verificación adicional (auto-recuperación): al usar el propio texto de un chunk como consulta, el motor lo recupera correctamente como resultado 1 con similitud 1.0 — el mecanismo de búsqueda/indexado es correcto. Sin embargo, con una consulta en lenguaje natural muy cercana al título del artículo ("Objeto de la Ley"), el modelo no ubica "Artículo 1. Objeto de la Ley" en el top-1 (recupera primero "Artículo 3. Ámbito de aplicación"), y para la pregunta "¿Cuál es el objeto de la Ley N.° 32069?" el chunk correcto quedó en el puesto 171 de 1358. Esto indica una limitación de calidad semántica del modelo de embeddings local elegido para este dominio (normativa legal en español), no un defecto del pipeline de chunking/indexado.

Recomendación: antes de usar este índice para responder preguntas reales, ejecutar la comparativa de embeddings prevista en la arquitectura del proyecto (`config.yaml -> evaluation.embedding_providers_to_compare`) contra al menos un proveedor adicional (p. ej. `openai` con `text-embedding-3-small`, que requiere `OPENAI_API_KEY`) y/o subir `retrieval.top_k` como mitigación parcial, antes de dar por buena la calidad de recuperación.
