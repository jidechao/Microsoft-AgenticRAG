from __future__ import annotations

import hashlib
import random
from typing import Protocol

from openai import OpenAI


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        pass


class FakeEmbeddingClient:
    def __init__(self, dims: int = 16) -> None:
        self.dims = dims

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)
            rng = random.Random(seed)
            vectors.append([rng.uniform(-1, 1) for _ in range(self.dims)])
        return vectors


class SiliconFlowEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
