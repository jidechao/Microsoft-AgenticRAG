from pathlib import Path

from agenticrag.embeddings import FakeEmbeddingClient
from agenticrag.models import DocumentChunk
from agenticrag.retriever import ChromaRetriever


def test_retriever_adds_and_queries_chunks(tmp_path):
    retriever = ChromaRetriever(
        persist_dir=tmp_path / "chroma",
        embedding_client=FakeEmbeddingClient(dims=8),
        collection_name="test",
    )
    chunks = [
        DocumentChunk("doc1", Path("docs/a.md"), "A", "md", 0, 0, 0, "alpha content"),
        DocumentChunk("doc2", Path("docs/b.md"), "B", "md", 0, 0, 0, "beta content"),
    ]

    retriever.reset()
    retriever.add_chunks(chunks)
    results = retriever.query("alpha", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.content in {"alpha content", "beta content"}
    assert results[0].score >= 0
