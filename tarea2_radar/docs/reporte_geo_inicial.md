# Reporte Inicial — Radar Geográfico (Tarea 2)

Generado: 2026-09-23T05:52:57.491372+00:00

## 1. Verificación del GeoJSON de departamentos

Sin problemas: FeatureCollection válido, proyección WGS84/CRS84, 25 departamentos del catálogo presentes 1:1, todas las geometrías son Polygon/MultiPolygon válidas según shapely. Ver `src/test_geo.py` para la verificación programática completa (corre independiente de este script).

Tabla ubigeo(INEI)<->departamento extraída de las 25 features del propio GeoJSON (columna `ubigeo_property` de config.yaml -> geo).

## 2. Integración con datos de contratación regional

3000 contratos normalizados agregados a nivel de departamento. 25/25 departamentos tienen al menos un proceso registrado en esta muestra.

Top 5 departamentos por monto adjudicado en esta muestra:

| Departamento | Monto adjudicado | N° procesos | N° proveedores únicos | N° entidades únicas |
|---|---|---|---|---|
| LIMA | S/ 3,121,315,387.16 | 902 | 317 | 233 |
| PIURA | S/ 1,085,080,631.84 | 86 | 32 | 35 |
| AYACUCHO | S/ 507,915,755.21 | 101 | 40 | 49 |
| SAN MARTIN | S/ 384,034,758.85 | 51 | 15 | 33 |
| CAJAMARCA | S/ 263,095,110.16 | 125 | 36 | 56 |

## 3. Modelo de relación Ubigeo/Departamento <-> métricas

`src/spatial_analysis.py::compute_departamento_metrics` agrega `contratos.parquet` por `departamento` (monto adjudicado total, número de procesos, proveedores únicos vía `nunique` sobre `proveedor`, entidades únicas vía `nunique` sobre `comprador`). `join_geo_metrics` hace un LEFT JOIN de la tabla ubigeo (25 filas, siempre completa, viene del GeoJSON) con esas métricas, dejando explícito con `tiene_datos=False` cualquier departamento sin procesos en la muestra actual, en vez de que desaparezca silenciosamente del mapa.
