"""Almacén vectorial simple (pickle + numpy) con similitud de coseno.

Se evita una dependencia nativa como FAISS para mantener el proyecto
portable; el volumen de chunks de dos normas legales es pequeño y una
búsqueda por fuerza bruta vectorizada es suficiente.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    def __init__(self):
        self.chunk_ids: list[str] = []
        self.metadata: list[dict] = []
        self.vectors: np.ndarray | None = None

    def add(self, chunks: list[dict], vectors: np.ndarray) -> int:
        """Idempotente: descarta chunk_ids ya presentes en el índice."""
        existing = set(self.chunk_ids)
        new_indices = [i for i, c in enumerate(chunks) if c["chunk_id"] not in existing]
        if not new_indices:
            return 0

        new_vectors = vectors[new_indices]
        self.vectors = new_vectors if self.vectors is None else np.vstack([self.vectors, new_vectors])
        for i in new_indices:
            chunk = chunks[i]
            self.chunk_ids.append(chunk["chunk_id"])
            self.metadata.append({k: v for k, v in chunk.items() if k != "texto"} | {"texto": chunk["texto"]})
        return len(new_indices)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        if self.vectors is None or len(self.chunk_ids) == 0:
            return []
        sims = cosine_similarity(query_vector.reshape(1, -1), self.vectors)[0]
        top_indices = np.argsort(-sims)[:top_k]
        return [
            {**self.metadata[i], "similitud": float(sims[i])}
            for i in top_indices
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "metadata": self.metadata, "vectors": self.vectors}, f)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        store = cls()
        if not path.exists():
            return store
        with path.open("rb") as f:
            data = pickle.load(f)
        store.chunk_ids = data["chunk_ids"]
        store.metadata = data["metadata"]
        store.vectors = data["vectors"]
        return store

    def __len__(self) -> int:
        return len(self.chunk_ids)
