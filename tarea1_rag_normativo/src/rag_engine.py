"""Motor RAG normativo. NO debe importar streamlit, telegram ni ninguna librería de UI.

Expone query(question, filters=None) -> dict con: answer, sources, abstained,
tokens_in, tokens_out, cost_usd, latency_sec, error.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .embeddings import build_embedding_provider
from .logger_cost import compute_cost_usd, log_query_cost
from .vector_store import VectorStore

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


class RagEngine:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.embedding_provider = build_embedding_provider(self.config["embeddings"])
        self.store = VectorStore.load(_resolve_path(self.config, "index_file"))
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
            if len(self.store) == 0:
                raise RuntimeError(
                    "El índice está vacío. Ejecuta build_index.py antes de iniciar la aplicación."
                )

            query_vector = self.embedding_provider.embed([question])[0]
            retrieval_cfg = self.config["retrieval"]
            top_k = retrieval_cfg["top_k"]
            candidates = self.store.search(query_vector, top_k * 3 if filters else top_k)

            documento_filtro = (filters or {}).get("documento_id")
            if documento_filtro:
                candidates = [c for c in candidates if c["documento_id"] == documento_filtro]
            candidates = candidates[:top_k]

            max_similarity = max((c["similitud"] for c in candidates), default=0.0)
            result["sources"] = [
                {
                    "documento": c["documento_titulo"],
                    "documento_id": c["documento_id"],
                    "version": c["version"],
                    "pagina": c["pagina"],
                    "similitud": round(c["similitud"], 4),
                }
                for c in candidates
            ]

            if max_similarity < retrieval_cfg["similarity_threshold"]:
                result["abstained"] = True
                result["answer"] = retrieval_cfg["abstention_message"]
                result["latency_sec"] = time.monotonic() - start
                log_query_cost(_resolve_path(self.config, "cost_log"), question, True, 0, 0, 0.0, result["latency_sec"])
                return result

            answer, tokens_in, tokens_out = self._generate_answer(question, candidates)
            llm_cfg = self.config["llm"]
            cost_usd = compute_cost_usd(
                tokens_in, tokens_out, llm_cfg["price_input_per_1k_usd"], llm_cfg["price_output_per_1k_usd"],
            )

            result.update({
                "answer": answer, "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": cost_usd,
            })
            result["latency_sec"] = time.monotonic() - start
            log_query_cost(
                _resolve_path(self.config, "cost_log"), question, False,
                tokens_in, tokens_out, cost_usd, result["latency_sec"],
            )
            return result
        except Exception as exc:  # noqa: BLE001 - se reporta estructuradamente, no se re-lanza a la UI
            result["error"] = str(exc)
            result["latency_sec"] = time.monotonic() - start
            return result

    def _generate_answer(self, question: str, candidates: list[dict]) -> tuple[str, int, int]:
        context_blocks = [
            f"[{c['documento_titulo']} v.{c['version']}, pág. {c['pagina']}]\n{c['texto']}"
            for c in candidates
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
                "content": f"Contexto:\n\n{context}\n\nPregunta: {question}",
            }],
        )
        answer_text = "".join(block.text for block in response.content if block.type == "text")
        return answer_text, response.usage.input_tokens, response.usage.output_tokens


def query(question: str, filters: dict | None = None) -> dict:
    """Función de conveniencia con motor cacheado a nivel de módulo."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RagEngine()
    return _ENGINE.query(question, filters)


_ENGINE: RagEngine | None = None
