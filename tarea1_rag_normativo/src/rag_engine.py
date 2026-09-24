"""Motor RAG normativo. NO debe importar streamlit, telegram ni ninguna librería de UI.

Expone query(question, filters=None) -> dict con: answer, sources, abstained,
tokens_in, tokens_out, cost_usd, latency_sec, error, citations_verified.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .embeddings import build_embedding_provider
from .logger_cost import compute_cost_usd, log_query_cost
from .vector_store import VectorStore

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR.parent / ".env"

_CITATION_ARTICULO_PATTERN = re.compile(r"art[íi]culo\s+\d+|art\.\s*\d+", re.IGNORECASE)
_CITATION_PAGE_PATTERN = re.compile(r"p[áa]g\.?\s*\d+|p[áa]gina\s+\d+", re.IGNORECASE)


def _has_explicit_citation(answer_text: str) -> bool:
    return bool(_CITATION_ARTICULO_PATTERN.search(answer_text)) and bool(_CITATION_PAGE_PATTERN.search(answer_text))


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
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY no está definido en .env")
            self._llm_client = genai.Client(api_key=api_key)
        return self._llm_client

    def query(self, question: str, filters: dict | None = None) -> dict:
        start = time.monotonic()
        result = {
            "answer": None, "sources": [], "abstained": False,
            "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
            "latency_sec": 0.0, "error": None, "citations_verified": None,
        }
        try:
            if len(self.store) == 0:
                raise RuntimeError(
                    "El índice está vacío. Ejecuta build_index.py antes de iniciar la aplicación."
                )

            query_vector = self.embedding_provider.embed_query(question)
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
                    "ley_ds": c.get("ley_ds"),
                    "version": c["version"],
                    "page_number": c["page_number"],
                    "titulo": c.get("titulo"),
                    "articulo": c.get("articulo"),
                    "numeral": c.get("numeral"),
                    "inciso": c.get("inciso"),
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

            answer, tokens_in, tokens_out, citations_verified = self._generate_answer(question, candidates)
            result["citations_verified"] = citations_verified
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

    def _generate_answer(self, question: str, candidates: list[dict]) -> tuple[str, int, int, bool]:
        """Genera la respuesta y VERIFICA que cite artículo+página de forma
        explícita (no confía ciegamente en que el LLM siga la instrucción del
        system prompt): si falta, reintenta una vez con un recordatorio
        explícito antes de devolver la respuesta final.
        """
        context_blocks = [
            f"[{c['documento_titulo']} v.{c['version']}, pág. {c['page_number']}"
            + (f", {c['titulo']}" if c.get("titulo") else "")
            + (f", {c['articulo']}" if c.get("articulo") else "")
            + (f", numeral {c['numeral']}" if c.get("numeral") else "")
            + (f", inciso {c['inciso']}" if c.get("inciso") else "")
            + f"]\n{c['texto']}"
            for c in candidates
        ]
        context = "\n\n---\n\n".join(context_blocks)
        llm_cfg = self.config["llm"]
        client = self._get_llm_client()
        generation_config = self._build_generation_config(llm_cfg)

        contents = [{"role": "user", "parts": [{"text": f"Contexto:\n\n{context}\n\nPregunta: {question}"}]}]
        response = client.models.generate_content(model=llm_cfg["model"], contents=contents, config=generation_config)
        answer_text, tokens_in, tokens_out = self._extract_text_and_usage(response)

        if _has_explicit_citation(answer_text):
            return answer_text, tokens_in, tokens_out, True

        contents.append({"role": "model", "parts": [{"text": answer_text}]})
        contents.append({"role": "user", "parts": [{"text": llm_cfg["citation_retry_message"]}]})
        retry_response = client.models.generate_content(model=llm_cfg["model"], contents=contents, config=generation_config)
        retry_text, retry_tokens_in, retry_tokens_out = self._extract_text_and_usage(retry_response)
        tokens_in += retry_tokens_in
        tokens_out += retry_tokens_out

        return retry_text, tokens_in, tokens_out, _has_explicit_citation(retry_text)

    @staticmethod
    def _build_generation_config(llm_cfg: dict):
        from google.genai import types
        return types.GenerateContentConfig(
            system_instruction=llm_cfg["system_prompt"],
            max_output_tokens=llm_cfg["max_tokens"],
            temperature=llm_cfg["temperature"],
            # Sin esto, gemini-3.6-flash puede gastar casi todo max_output_tokens
            # "pensando" y truncar la respuesta visible a media frase (medido en
            # vivo). thinking_budget=0 fuerza una respuesta directa y completa.
            thinking_config=types.ThinkingConfig(thinking_budget=llm_cfg.get("thinking_budget", 0)),
        )

    @staticmethod
    def _extract_text_and_usage(response) -> tuple[str, int, int]:
        text = response.text or ""
        usage = response.usage_metadata
        tokens_in = usage.prompt_token_count or 0
        tokens_out = usage.candidates_token_count or 0
        return text, tokens_in, tokens_out


def query(question: str, filters: dict | None = None) -> dict:
    """Función de conveniencia con motor cacheado a nivel de módulo."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RagEngine()
    return _ENGINE.query(question, filters)


_ENGINE: RagEngine | None = None
