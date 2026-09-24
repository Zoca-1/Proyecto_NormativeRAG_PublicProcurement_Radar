# Proyecto Integrador: RAG Normativo y Radar de Contrataciones Públicas

Dos herramientas RAG sobre contrataciones públicas en Perú:

- **Tarea 1 — RAG Normativo**: asistente legal sobre la Ley N.° 32069 y el D.S. N.° 001-2026-EF, con fragmentación guiada por estructura legal (Título/Artículo/Numeral/Inciso), abstención por umbral de similitud y citas trazables por artículo, numeral, inciso, página y versión.
- **Tarea 2 — Radar de Contrataciones**: pipeline de datos abiertos OECE (OCDS), RAG híbrido (filtros estructurados + embeddings), indicador de riesgo de pujador único y dashboard con mapa coroplético.

## Arquitectura (reglas inviolables)

1. **Motor desacoplado de la UI.** `tarea1_rag_normativo/src/rag_engine.py` y `tarea2_radar/src/hybrid_engine.py` no importan `streamlit` ni ninguna librería de interfaz. Cada uno expone `query(question, filters=None) -> dict` con: `answer`, `sources`, `abstained`, `tokens_in`, `tokens_out`, `cost_usd`, `latency_sec`, `error` (Tarea 1 agrega `citations_verified`).
2. **Cero valores hardcodeados.** Todo parámetro vive en `config.yaml` de cada tarea; credenciales en `.env` (ver `.env.example`).
3. **Offline vs online.** `build_index.py` (offline, idempotente y reanudable) construye los índices. `app.py` (online, Streamlit) solo lee lo ya construido y llama al motor — nunca extrae ni indexa al iniciar.
4. **Trazabilidad.** Tarea 1 conserva `documento_id`/`documento_titulo`/`version`/`page_number` desde la extracción del PDF (nunca fusiona páginas en un solo string antes de fragmentar) y agrega `titulo`/`articulo`/`numeral`/`inciso` al fragmentar por estructura legal. Tarea 2 conserva `ocid`, `departamento`, `monto`, `fecha`, `categoria`, `comprador` en cada fragmento.
5. **Abstención estructurada.** La similitud máxima se compara contra `retrieval.similarity_threshold` **antes** de llamar al LLM. Si no alcanza el umbral, `abstained=true` y no se gasta ningún token de LLM. Cada consulta con LLM se registra en `logs/cost_log.csv`.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # completar GEMINI_API_KEY (Tarea 1), ANTHROPIC_API_KEY (Tarea 2), etc.
```

## Tarea 1 — RAG Normativo

1. Coloca los PDFs (`LEY_32069.pdf`, `DS_001_2026_EF.pdf`) en `tarea1_rag_normativo/data/raw/`, según los nombres declarados en `tarea1_rag_normativo/config.yaml -> documents`.
2. Inspecciona las fuentes, extrae con `page_number` preservado y limpia el texto (genera `docs/reporte_extraccion.md`):
   ```bash
   python tarea1_rag_normativo/source_check.py
   ```
3. (Opcional) Compara proveedores de embeddings antes de indexar — corre en memoria, no toca el índice de producción, y registra Recall@k/MRR en `eval/resultados.csv`:
   ```bash
   python tarea1_rag_normativo/eval/evaluate_retrieval.py
   ```
   Ajusta `config.yaml -> embeddings` al candidato ganador antes del siguiente paso.
4. Construye el índice (offline, idempotente; fragmenta por Título/Artículo/Numeral/Inciso y genera `docs/reporte_indexacion.md`, que incluye el Recall@k del proveedor configurado):
   ```bash
   python tarea1_rag_normativo/build_index.py
   ```
   Usa `--force` para reconstruir desde cero (necesario tras cambiar el modelo de embeddings).
5. Lanza la app (solo lectura del índice):
   ```bash
   streamlit run tarea1_rag_normativo/app.py
   ```

## Tarea 2 — Radar de Contrataciones

1. Configura `tarea2_radar/config.yaml -> oece` con la URL real del portal de datos abiertos OECE (OCDS 2026).
2. Ejecuta el pipeline offline (fetch → normalización → riesgo → índice híbrido):
   ```bash
   python tarea2_radar/build_index.py
   ```
   Reanudable: si se interrumpe, vuelve a ejecutarse y continúa desde el último cursor guardado en `data/raw/_fetch_state.json`.
3. Coloca un GeoJSON de los 25 departamentos del Perú en la ruta indicada por `config.yaml -> geo.geojson_departamentos` para el mapa coroplético.
4. Lanza el dashboard:
   ```bash
   streamlit run tarea2_radar/app.py
   ```

## Verificar el desacoplamiento motor/UI

```bash
grep -niE "streamlit|telegram" tarea1_rag_normativo/src/rag_engine.py tarea2_radar/src/hybrid_engine.py
```
No debe imprimir ninguna coincidencia.

## Estructura

Ver el árbol completo en `docs/pipeline.md`, que además documenta el flujo offline/online de ambas tareas.

## 📹 Video de Demostración y Presentación Técnica

El video explicativo de 12 minutos (que cubre la arquitectura *pipeline-first*, decisiones técnicas, demostración en vivo de ambas aplicaciones y el desglose de costos) se encuentra disponible en Google Drive:

👉 **[Ver Video de Demostración del Proyecto en Google Drive](https://drive.google.com/drive/u/1/folders/1ZTR5nS8BYiUhD-MUT9DNTFSRQx8O8Azn)**






