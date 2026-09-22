"""Motor RAG híbrido. NO debe importar streamlit, telegram ni ninguna librería de UI.

Expone query(question, filters=None) -> dict con: answer, sources, abstained,
tokens_in, tokens_out, cost_usd, latency_sec, error.

filters admite: departamento, categoria, monto_min, monto_max, fecha_desde, fecha_hasta.
El filtrado estructurado se aplica ANTES de la búsqueda por similitud (RAG híbrido).
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from .hybrid_indexer import HybridIndex, build_embedding_provider

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR.parent / ".env"


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or (BASE_DIR / "config.yaml")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    load_dotenv(ENV_PATH)
    return config


def _resolve_path(config: dict, key: str) -> Path:
    return BASE_DIR / config["paths"][key]


def _apply_structured_filters(metadata: pd.DataFrame, filters: dict) -> pd.DataFrame:
    df = metadata
    if departamento := filters.get("departamento"):
        df = df[df["departamento"] == departamento]
    if categoria := filters.get("categoria"):
        df = df[df["categoria"] == categoria]
    if (monto_min := filters.get("monto_min")) is not None:
        df = df[df["monto"] >= monto_min]
    if (monto_max := filters.get("monto_max")) is not None:
        df = df[df["monto"] <= monto_max]
    if fecha_desde := filters.get("fecha_desde"):
        df = df[df["fecha"] >= pd.Timestamp(fecha_desde, tz="UTC")]
    if fecha_hasta := filters.get("fecha_hasta"):
        df = df[df["fecha"] <= pd.Timestamp(fecha_hasta, tz="UTC")]
    return df


_COST_LOG_FIELDNAMES = [
    "timestamp_utc", "pregunta", "abstained", "tokens_in", "tokens_out",
    "cost_usd", "latency_sec",
]


def _log_query_cost(cost_log_path: Path, pregunta: str, abstained: bool, tokens_in: int, tokens_out: int,
                     cost_usd: float, latency_sec: float) -> None:
    cost_log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not cost_log_path.exists()
    with cost_log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COST_LOG_FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pregunta": pregunta,
            "abstained": abstained,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost_usd, 6),
            "latency_sec": round(latency_sec, 3),
        })


class HybridEngine:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.embedding_provider = build_embedding_provider(self.config["embeddings"])
        self.index = HybridIndex.load(_resolve_path(self.config, "index_file"))
        self._llm_client = None

    def _get_llm_client(self):
        if self._llm_client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY no está definido en .env")
            self._llm_client = anthropic.Anthropic(api_key=api_key)
        return self._llm_client

    def query(self, question: str, filters: dict | None = None) -> dict:
        start = time.monotonic()
        result = {
            "answer": None, "sources": [], "abstained": False,
            "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
            "latency_sec": 0.0, "error": None,
        }
        try:
            if len(self.index) == 0:
                raise RuntimeError(
                    "El índice está vacío. Ejecuta build_index.py antes de iniciar la aplicación."
                )

            candidates_df = _apply_structured_filters(self.index.metadata, filters or {})
            retrieval_cfg = self.config["retrieval"]
            top_k = retrieval_cfg["top_k"]

            if candidates_df.empty:
                result["abstained"] = True
                result["answer"] = retrieval_cfg["abstention_message"]
                result["latency_sec"] = time.monotonic() - start
                _log_query_cost(_resolve_path(self.config, "cost_log"), question, True, 0, 0, 0.0, result["latency_sec"])
                return result

            candidate_indices = candidates_df.index.to_numpy()
            candidate_vectors = self.index.vectors[candidate_indices]

            query_vector = self.embedding_provider.embed([question])[0]
            sims = cosine_similarity(query_vector.reshape(1, -1), candidate_vectors)[0]
            top_order = np.argsort(-sims)[:top_k]

            top_rows = candidates_df.iloc[top_order]
            top_sims = sims[top_order]
            max_similarity = float(top_sims.max()) if len(top_sims) else 0.0

            result["sources"] = [
                {
                    "ocid": row["ocid"],
                    "departamento": row["departamento"],
                    "monto": float(row["monto"]),
                    "comprador": row["comprador"],
                    "categoria": row["categoria"],
                    "similitud": round(float(sim), 4),
                }
                for (_, row), sim in zip(top_rows.iterrows(), top_sims)
            ]

            if max_similarity < retrieval_cfg["similarity_threshold"]:
                result["abstained"] = True
                result["answer"] = retrieval_cfg["abstention_message"]
                result["latency_sec"] = time.monotonic() - start
                _log_query_cost(_resolve_path(self.config, "cost_log"), question, True, 0, 0, 0.0, result["latency_sec"])
                return result

            answer, tokens_in, tokens_out = self._generate_answer(question, top_rows)
            llm_cfg = self.config["llm"]
            cost_usd = (
                (tokens_in / 1000) * llm_cfg["price_input_per_1k_usd"]
                + (tokens_out / 1000) * llm_cfg["price_output_per_1k_usd"]
            )

            result.update({
                "answer": answer, "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": cost_usd,
            })
            result["latency_sec"] = time.monotonic() - start
            _log_query_cost(
                _resolve_path(self.config, "cost_log"), question, False,
                tokens_in, tokens_out, cost_usd, result["latency_sec"],
            )
            return result
        except Exception as exc:  # noqa: BLE001 - se reporta estructuradamente, no se re-lanza a la UI
            result["error"] = str(exc)
            result["latency_sec"] = time.monotonic() - start
            return result

    def _generate_answer(self, question: str, top_rows: pd.DataFrame) -> tuple[str, int, int]:
        context_blocks = [
            f"[OCID {row['ocid']} | {row['departamento']} | S/ {row['monto']:,.2f} | {row['categoria']}]\n"
            f"Comprador: {row['comprador']}\nObjeto: {row.get('objeto_contratacion', '')}"
            for _, row in top_rows.iterrows()
        ]
        context = "\n\n---\n\n".join(context_blocks)
        llm_cfg = self.config["llm"]

        client = self._get_llm_client()
        response = client.messages.create(
            model=llm_cfg["model"],
            max_tokens=llm_cfg["max_tokens"],
            temperature=llm_cfg["temperature"],
            system=llm_cfg["system_prompt"],
            messages=[{
                "role": "user",
                "content": f"Contratos relevantes:\n\n{context}\n\nPregunta: {question}",
            }],
        )
        answer_text = "".join(block.text for block in response.content if block.type == "text")
        return answer_text, response.usage.input_tokens, response.usage.output_tokens


def query(question: str, filters: dict | None = None) -> dict:
    """Función de conveniencia con motor cacheado a nivel de módulo."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = HybridEngine()
    return _ENGINE.query(question, filters)


_ENGINE: HybridEngine | None = None
