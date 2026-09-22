"""Script OFFLINE: fetch OECE -> normalización -> indicador de riesgo -> índice híbrido.

Idempotente y reanudable: el fetch retoma desde _fetch_state.json, y cada
etapa posterior se puede re-ejecutar sin duplicar filas (dedupe por ocid).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.hybrid_engine import load_config
from src.hybrid_indexer import HybridIndex, build_embedding_provider, build_texto_para_embedding
from src.oece_fetcher import fetch_releases
from src.risk_analyzer import compute_risk_by_categoria, compute_risk_indicators
from src.validator_normalizer import load_and_normalize, save_rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el pipeline de datos y el índice híbrido de Tarea 2.")
    parser.add_argument("--refresh", action="store_true", help="Reinicia el estado de paginación y re-descarga todo.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config = load_config()

    raw_releases_path = base_dir / config["paths"]["raw_releases_file"]
    state_path = base_dir / config["paths"]["fetch_state_file"]
    contracts_path = base_dir / config["paths"]["contracts_parquet"]
    risk_path = base_dir / config["paths"]["risk_parquet"]
    index_path = base_dir / config["paths"]["index_file"]
    rejected_path = base_dir / config["normalizacion"]["rejected_log_file"]

    if args.refresh:
        raw_releases_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    new_releases = fetch_releases(config["oece"], raw_releases_path, state_path)
    print(f"[1/5] Fetch OECE: {new_releases} releases nuevos descargados")

    if not raw_releases_path.exists():
        print("No hay datos crudos descargados todavía. Verifica config.yaml -> oece.")
        return

    df, rejected = load_and_normalize(raw_releases_path, config["departamentos"], config["normalizacion"])
    save_rejected(rejected, rejected_path)
    contracts_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(contracts_path, index=False)
    print(f"[2/5] Normalización: {len(df)} contratos válidos, {len(rejected)} filas rechazadas")

    risk_df = compute_risk_indicators(df, config["riesgo"])
    risk_df.to_parquet(risk_path, index=False)
    compute_risk_by_categoria(df, config["riesgo"])
    print(f"[3/5] Indicador de riesgo: {len(risk_df)} departamentos con métricas de pujador único")

    index = HybridIndex.load(index_path)
    existing_ocids = set(index.metadata["ocid"]) if not index.metadata.empty else set()
    pending_df = df[~df["ocid"].isin(existing_ocids)].reset_index(drop=True)

    if pending_df.empty:
        print("[4/5] Embeddings: ok (índice ya actualizado)")
    else:
        provider = build_embedding_provider(config["embeddings"])
        campos = config["campos_texto_para_embedding"]
        textos = pending_df.apply(lambda row: build_texto_para_embedding(row, campos), axis=1).tolist()
        vectors = provider.embed(textos)
        added = index.add(pending_df, vectors)
        index.save(index_path)
        print(f"[4/5] Embeddings: {added} vectores nuevos agregados al índice híbrido")

    print(f"[5/5] Índice final: {len(index)} contratos indexados en {index_path}")


if __name__ == "__main__":
    main()
