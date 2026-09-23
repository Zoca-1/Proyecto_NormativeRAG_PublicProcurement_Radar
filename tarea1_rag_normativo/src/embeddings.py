"""Proveedores de embeddings intercambiables (comparativa costo-cero vs. pago).

Los modelos de la familia E5 (intfloat/multilingual-e5-*) requieren anteponer
"query: " a las consultas y "passage: " a los pasajes indexados para rendir
como fueron entrenados; otros modelos (MiniLM, BGE sin instrucción) usan
prefijos vacíos. Por eso embed_query/embed_passages están separados en la
interfaz en vez de un único embed(texts) genérico, y los prefijos vienen de
config.yaml (embeddings.query_prefix/passage_prefix), nunca hardcodeados.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    name: str

    def __init__(self, query_prefix: str = "", passage_prefix: str = ""):
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    @abstractmethod
    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        ...

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed_raw([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_raw([self.query_prefix + text])[0]


class LocalEmbeddingProvider(EmbeddingProvider):
    name = "local"

    def __init__(self, model_name: str, batch_size: int = 16, query_prefix: str = "", passage_prefix: str = ""):
        super().__init__(query_prefix, passage_prefix)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, convert_to_numpy=True, normalize_embeddings=True,
        )
        return vectors.astype(np.float32)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, model_name: str, batch_size: int = 16, query_prefix: str = "", passage_prefix: str = ""):
        super().__init__(query_prefix, passage_prefix)
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no está definido en .env")
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._batch_size = batch_size

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            response = self._client.embeddings.create(model=self._model_name, input=batch)
            vectors.extend(item.embedding for item in response.data)
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def build_embedding_provider(embeddings_config: dict) -> EmbeddingProvider:
    provider = embeddings_config["provider"]
    batch_size = embeddings_config.get("batch_size", 16)
    query_prefix = embeddings_config.get("query_prefix", "")
    passage_prefix = embeddings_config.get("passage_prefix", "")
    if provider == "local":
        return LocalEmbeddingProvider(embeddings_config["local_model"], batch_size, query_prefix, passage_prefix)
    if provider == "openai":
        return OpenAIEmbeddingProvider(embeddings_config["openai_model"], batch_size, query_prefix, passage_prefix)
    raise ValueError(f"Proveedor de embeddings no soportado: {provider}")
