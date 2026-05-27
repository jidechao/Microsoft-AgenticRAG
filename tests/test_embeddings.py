from agenticrag.embeddings import FakeEmbeddingClient


def test_fake_embedding_is_deterministic():
    client = FakeEmbeddingClient(dims=4)

    first = client.embed_texts(["alpha", "beta"])
    second = client.embed_texts(["alpha", "beta"])

    assert first == second
    assert len(first) == 2
    assert len(first[0]) == 4
    assert first[0] != first[1]
