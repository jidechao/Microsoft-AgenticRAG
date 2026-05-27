from types import SimpleNamespace

import pytest

from agenticrag.embeddings import FakeEmbeddingClient
from agenticrag.embeddings import SiliconFlowEmbeddingClient


def test_fake_embedding_is_deterministic():
    client = FakeEmbeddingClient(dims=4)

    first = client.embed_texts(["alpha", "beta"])
    second = client.embed_texts(["alpha", "beta"])

    assert first == second
    assert len(first) == 2
    assert len(first[0]) == 4
    assert first[0] != first[1]


def test_fake_embedding_rejects_invalid_dims():
    with pytest.raises(ValueError, match="dims must be greater than 0"):
        FakeEmbeddingClient(dims=0)


def make_siliconflow_client(data):
    client = SiliconFlowEmbeddingClient.__new__(SiliconFlowEmbeddingClient)
    client.model = "test-model"
    client.client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda model, input: SimpleNamespace(data=data),
        ),
    )
    return client


def test_siliconflow_embedding_rejects_cardinality_mismatch():
    client = make_siliconflow_client(
        [
            SimpleNamespace(embedding=[1.0], index=0),
        ],
    )

    with pytest.raises(ValueError, match="Expected 2 embeddings, received 1 embeddings"):
        client.embed_texts(["alpha", "beta"])


def test_siliconflow_embedding_sorts_by_index_when_present():
    client = make_siliconflow_client(
        [
            SimpleNamespace(embedding=[2.0], index=1),
            SimpleNamespace(embedding=[1.0], index=0),
        ],
    )

    assert client.embed_texts(["alpha", "beta"]) == [[1.0], [2.0]]
