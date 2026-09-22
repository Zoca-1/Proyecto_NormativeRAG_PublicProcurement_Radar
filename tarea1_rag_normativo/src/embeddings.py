"""Proveedores de embeddings intercambiables (comparativa costo-cero vs. pago).

El proveedor "local" (sentence-transformers) no consume API y sirve como
baseline gratuito para la comparativa exigida en la evaluación.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    name = "local"

    def __init__(self, model_name: str, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, convert_to_numpy=True, normalize_embeddings=True,
        )
        return vectors.astype(np.float32)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, model_name: str, batch_size: int = 16):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no está definido en .env")
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
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
    if provider == "local":
        return LocalEmbeddingProvider(embeddings_config["local_model"], batch_size)
    if provider == "openai":
        return OpenAIEmbeddingProvider(embeddings_config["openai_model"], batch_size)
    raise ValueError(f"Proveedor de embeddings no soportado: {provider}")
