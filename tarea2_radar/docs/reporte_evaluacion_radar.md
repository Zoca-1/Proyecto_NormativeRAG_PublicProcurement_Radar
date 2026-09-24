# Reporte de Evaluación — Radar de Contrataciones (Tarea 2)

Generado: 2026-09-23T06:00:25.494110+00:00

## 1. Cobertura de los 25 departamentos

- En `contratos.parquet`: **25/25** (100%).
- En el índice híbrido (`hybrid_index.pkl`): **25/25** (100%).
- Con resultados reales de retrieval (filtro + similitud, no solo presencia en el catálogo): **25/25** (100%).

## 2. Retrieval híbrido (filtros estructurados + embeddings)

- Preguntas evaluadas: 25 (una por departamento, con su categoría más frecuente en los datos reales).
- Con al menos un candidato tras el filtro estructurado: 25/25 (100%).
- Con similitud top-1 ≥ umbral de abstención (0.3): 25/25 (100%).
- Similitud top-1 promedio: 0.5493.

Detalle completo en `eval/resultados_radar.csv`.

## 3. Precisión/consistencia del índice de riesgo (pujador único)

No existe una etiqueta externa de "riesgo real" contra la cual medir precisión en el sentido de clasificación supervisada (no hay un ground truth de qué contratos son efectivamente irregulares); lo que se puede y se verifica aquí es la **correctitud aritmética**: recalcular el indicador desde `contratos.parquet` con `risk_analyzer.compute_risk_indicators` y compararlo fila por fila contra `riesgo_departamento.parquet` ya persistido.

Sin discrepancias: las 25 filas (una por departamento) de `riesgo_departamento.parquet` coinciden exactamente con el recálculo desde los contratos actuales — el indicador persistido está al día y es aritméticamente correcto.

Hallazgo: 1 de 3000 filas (0.03%) en `contratos.parquet` comparten `ocid` con otra fila (ejemplo: `ocds-dgv273-seacev3-2026-2543-268`). Esto es consistente con el comportamiento real de la API de OECE: un mismo procedimiento puede publicar releases separados por etapa del ciclo de vida (planning/tender, award, contract), cada uno como una fila independiente. Efecto actual mínimo, pero al aumentar `oece.max_pages` la tasa de duplicación podría crecer — si se necesita 'número de procedimientos únicos' en vez de 'número de releases', deduplicar por `ocid` antes de agregar.

## 4. Índices de concentración

- HHI de participación de mercado por proveedor (nacional, por monto adjudicado): **718.5** (mercado no concentrado, escala estándar 0–10000: <1500 no concentrado, 1500–2500 moderado, >2500 alta concentración).
- Los 5 departamentos de mayor monto adjudicado concentran el **74.9%** del monto total adjudicado en la muestra.

| Departamento | Monto adjudicado |
|---|---|
| LIMA | S/ 3,121,315,387.16 |
| PIURA | S/ 1,085,080,631.84 |
| AYACUCHO | S/ 507,915,755.21 |
| SAN MARTIN | S/ 384,034,758.85 |
| CAJAMARCA | S/ 263,095,110.16 |
