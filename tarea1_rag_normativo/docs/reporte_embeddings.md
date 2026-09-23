# Reporte de Comparativa de Embeddings — Tarea 1 (RAG Normativo)

## 1. Contexto y por qué se hizo esta comparativa

El primer índice de producción (Fase 2) usaba
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` como baseline
costo-cero. Al construirlo se detectaron dos problemas que motivaron esta
comparativa:

1. El set de evaluación original (`eval/preguntas.csv`) tenía solo 5
   preguntas con páginas esperadas *adivinadas* antes de tener acceso a los
   PDFs reales, y el emparejamiento era por `page_number` exacto — muy frágil
   frente a un chunking por estructura legal donde un mismo artículo puede
   quedar en varios fragmentos. Se reconstruyó con **26 preguntas verificadas
   manualmente contra el contenido real** de ambos documentos, y el
   emparejamiento pasó a ser por `documento_id` + prefijo de `articulo`
   (más robusto a la granularidad del chunking).
2. Con esa metodología corregida, el propio baseline MiniLM ya rendía
   razonablemente (76.9% recall@5), pero no alcanzaba la meta de 80% fijada
   en `config.yaml -> evaluation.target_recall_at_k`.

## 2. Candidatos evaluados

`OPENAI_API_KEY` no está configurado en este entorno (`.env` vacío para esa
clave), así que `text-embedding-3-small` no pudo evaluarse vía API. En su
lugar se compararon dos modelos multilingües locales de mayor capacidad que
el baseline, ambos costo-cero (corren en CPU, sin llamadas externas):

| Candidato | Modelo | Dim. | max_seq_length | Prefijos requeridos |
|---|---|---|---|---|
| `local_minilm_baseline` | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 128 subtokens | ninguno |
| `local_multilingual_e5_base` | `intfloat/multilingual-e5-base` | 768 | 512 subtokens | `query: ` / `passage: ` (obligatorios; el modelo se entrenó con ellos) |
| `local_bge_m3` | `BAAI/bge-m3` | 1024 | 8192 subtokens | ninguno (probado sin instrucción) |

`src/embeddings.py` se generalizó para soportar `query_prefix`/`passage_prefix`
por proveedor (`embed_query` vs. `embed_passages`), ya que los modelos E5 rinden
notablemente peor sin el prefijo correcto.

## 3. Metodología

`eval/evaluate_retrieval.py`, para cada candidato:
1. Construye el proveedor de embeddings y **embebe todo el corpus de
   `chunks.jsonl` (1358 fragmentos) en memoria** — no toca
   `data/processed/vector_index.pkl` (el índice de producción), así la
   comparativa no interfiere con lo que esté desplegado.
2. Embebe cada una de las 26 preguntas como consulta (`embed_query`, con el
   prefijo del candidato) y rankea **todo** el corpus (no solo el top-k).
3. Un acierto de Recall@5 es que el chunk correcto (mismo `documento_id`,
   `articulo` con el prefijo esperado) aparezca en el top-5. El MRR usa el
   rango real del primer acierto en todo el corpus (0 si no aparece nunca).
4. Nada de esto llama al LLM: comparación 100% costo cero.

## 4. Resultados

| Candidato | Recall@5 | MRR | Tiempo embeber corpus (CPU) | ¿Cumple meta 80%? |
|---|---|---|---|---|
| `local_minilm_baseline` | 76.9% (20/26) | 0.5761 | 14.8 s | ❌ No |
| `local_multilingual_e5_base` | 84.6% (22/26) | 0.7667 | 60.2 s | ✅ Sí |
| **`local_bge_m3`** | **92.3% (24/26)** | **0.8463** | 213.6 s | ✅ Sí |

Detalle por pregunta en `eval/resultados.csv`.

### Preguntas que cada modelo falló

- **Baseline (6 fallos):** finalidad de la ley, causales de nulidad de actos
  procedimentales, nulidad de contrato por incumplir el procedimiento,
  pretensiones no arbitrables, quién sanciona infracciones, quién refrenda el D.S.
- **multilingual-e5-base (4 fallos):** objeto de la ley, ámbito de aplicación,
  causales de nulidad de actos procedimentales, dónde se publica el D.S.
- **bge-m3 (2 fallos, ambos también fallados por e5-base):** "¿Cuál es el
  objeto de la Ley N.° 32069?" y "¿Cuál es el ámbito de aplicación de la Ley
  N.° 32069?" — los Artículos 1 y 3 son encabezados cortos y temáticamente
  parecidos a otros artículos definitorios cortos de la misma norma; incluso
  el mejor modelo los confunde. Mitigación disponible si esto importa en
  producción: subir `retrieval.top_k` (hoy 5) para estas consultas de tipo
  "objeto/ámbito", o añadir un paso de expansión de consulta.

## 5. Decisión

**Se seleccionó `BAAI/bge-m3`** (mejor Recall@5 y MRR de los tres, con margen
sobre la meta de 80%) como modelo de producción en
`tarea1_rag_normativo/config.yaml -> embeddings`. El costo es un tiempo de
embebido de corpus ~3.6x mayor que e5-base y ~14x mayor que el baseline
(213.6s vs. 60.2s vs. 14.8s para 1358 fragmentos) — aceptable porque el
embebido del corpus es un paso *offline* de `build_index.py`, no afecta la
latencia de consulta en línea (una sola pregunta se embebe en milisegundos).

El índice de producción (`data/processed/vector_index.pkl`) ya fue
reconstruido con `bge-m3` (1358 chunks, dim=1024) y su Recall@5/MRR medido
directamente sobre ese índice (92.3%/0.8463, ver
`docs/reporte_indexacion.md`) coincide exactamente con esta comparativa.

## 6. Nota sobre `OPENAI_API_KEY`

Si se agrega `OPENAI_API_KEY` a `.env` más adelante, se puede repetir esta
comparativa agregando un candidato `provider: "openai"` /
`model: "text-embedding-3-small"` a `config.yaml ->
evaluation.embedding_candidates` y correr de nuevo
`eval/evaluate_retrieval.py`, sin tocar el índice de producción hasta decidir
si conviene cambiar.
