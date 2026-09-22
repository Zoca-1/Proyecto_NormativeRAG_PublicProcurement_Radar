"""Índice híbrido: metadatos estructurados (ocid, departamento, monto, fecha,
categoria, comprador) + embeddings del texto del contrato, en un único
almacén que permite filtrar primero y buscar por similitud después.

También aloja el proveedor de embeddings (local costo-cero u OpenAI), para
que build_index.py (offline) y hybrid_engine.py (online) usen la misma
implementación sin acoplar Tarea 2 al paquete de Tarea 1.
"""
from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class EmbeddingProvider(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    name = "local"

    def __init__(self, model_name: str, batch_size: int = 32):
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

    def __init__(self, model_name: str, batch_size: int = 32):
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
    batch_size = embeddings_config.get("batch_size", 32)
    if provider == "local":
        return LocalEmbeddingProvider(embeddings_config["local_model"], batch_size)
    if provider == "openai":
        return OpenAIEmbeddingProvider(embeddings_config["openai_model"], batch_size)
    raise ValueError(f"Proveedor de embeddings no soportado: {provider}")


class HybridIndex:
    def __init__(self):
        self.metadata: pd.DataFrame = pd.DataFrame()
        self.vectors: np.ndarray | None = None

    def add(self, metadata_df: pd.DataFrame, vectors: np.ndarray) -> int:
        """Idempotente: descarta ocids ya presentes en el índice."""
        existing_ocids = set(self.metadata["ocid"]) if not self.metadata.empty else set()
        mask_new = ~metadata_df["ocid"].isin(existing_ocids)
        if not mask_new.any():
            return 0

        new_metadata = metadata_df[mask_new].reset_index(drop=True)
        new_vectors = vectors[mask_new.to_numpy()]

        self.metadata = pd.concat([self.metadata, new_metadata], ignore_index=True)
        self.vectors = new_vectors if self.vectors is None else np.vstack([self.vectors, new_vectors])
        return len(new_metadata)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"metadata": self.metadata, "vectors": self.vectors}, f)

    @classmethod
    def load(cls, path: Path) -> "HybridIndex":
        index = cls()
        if not path.exists():
            return index
        with path.open("rb") as f:
            data = pickle.load(f)
        index.metadata = data["metadata"]
        index.vectors = data["vectors"]
        return index

    def __len__(self) -> int:
        return 0 if self.metadata.empty else len(self.metadata)


def build_texto_para_embedding(row: pd.Series, campos: list[str]) -> str:
    return " | ".join(str(row[campo]) for campo in campos if campo in row and pd.notna(row[campo]))
