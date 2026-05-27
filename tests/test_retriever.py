from __future__ import annotations

from pathlib import Path

import chromadb
import pytest

from agenticrag.embeddings import FakeEmbeddingClient
from agenticrag.models import DocumentChunk
from agenticrag.retriever import ChromaRetriever


class StubCollection:
    def __init__(self, query_response: dict | None = None) -> None:
        self.query_response = query_response or {}
        self.add_calls: list[dict] = []
        self.query_calls: list[dict] = []

    def add(self, *, ids, documents, embeddings, metadatas) -> None:
        self.add_calls.append(
            {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        )

    def query(self, *, query_embeddings, n_results, include) -> dict:
        self.query_calls.append(
            {
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "include": include,
            }
        )
        return self.query_response


class StubClient:
    def __init__(
        self,
        collection: StubCollection,
        delete_exception: Exception | None = None,
    ) -> None:
        self.collection = collection
        self.delete_exception = delete_exception
        self.deleted_names: list[str] = []
        self.created_names: list[str] = []

    def delete_collection(self, name: str) -> None:
        self.deleted_names.append(name)
        if self.delete_exception is not None:
            raise self.delete_exception

    def get_or_create_collection(self, name: str) -> StubCollection:
        self.created_names.append(name)
        return self.collection


def make_chunk(
    doc_id: str = "doc-1",
    path: str = "docs/a.md",
    title: str = "A",
    filetype: str = "md",
    chunk_index: int = 0,
    line_start: int = 0,
    line_end: int = 0,
    content: str = "alpha content",
) -> DocumentChunk:
    return DocumentChunk(
        doc_id,
        Path(path),
        title,
        filetype,
        chunk_index,
        line_start,
        line_end,
        content,
    )


def make_stubbed_retriever(
    query_response: dict | None = None,
    delete_exception: Exception | None = None,
) -> tuple[ChromaRetriever, StubCollection, StubClient]:
    collection = StubCollection(query_response=query_response)
    client = StubClient(collection=collection, delete_exception=delete_exception)
    retriever = ChromaRetriever.__new__(ChromaRetriever)
    retriever.persist_dir = Path("unused")
    retriever.embedding_client = FakeEmbeddingClient(dims=8)
    retriever.collection_name = "test"
    retriever.client = client
    retriever.collection = collection
    return retriever, collection, client


def test_retriever_adds_and_queries_chunks(tmp_path):
    retriever = ChromaRetriever(
        persist_dir=tmp_path / "chroma",
        embedding_client=FakeEmbeddingClient(dims=8),
        collection_name="test",
    )
    chunks = [
        make_chunk(doc_id="doc1", path="docs/a.md", title="A", content="alpha content"),
        make_chunk(doc_id="doc2", path="docs/b.md", title="B", content="beta content"),
    ]

    retriever.reset()
    retriever.add_chunks(chunks)
    results = retriever.query("alpha", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.content in {"alpha content", "beta content"}
    assert results[0].score >= 0


def test_add_chunks_with_empty_list_is_a_no_op():
    retriever, collection, _ = make_stubbed_retriever()

    retriever.add_chunks([])

    assert collection.add_calls == []


def test_query_returns_empty_list_for_empty_chroma_response():
    retriever, _, _ = make_stubbed_retriever(query_response={"documents": [], "metadatas": [], "distances": []})

    assert retriever.query("alpha", top_k=1) == []


def test_query_raises_for_misaligned_chroma_arrays():
    retriever, _, _ = make_stubbed_retriever(
        query_response={
            "documents": [["alpha content", "beta content"]],
            "metadatas": [[make_chunk(doc_id="doc1").to_metadata()]],
            "distances": [[0.1, 0.2]],
        }
    )

    with pytest.raises(ValueError, match="Chroma query response has misaligned result lengths"):
        retriever.query("alpha", top_k=2)


def test_query_reconstructs_chunk_metadata_correctly():
    chunk = make_chunk(
        doc_id="doc-9",
        path="docs/guide.md",
        title="Guide",
        filetype="markdown",
        chunk_index=4,
        line_start=12,
        line_end=18,
        content="stored separately",
    )
    retriever, _, _ = make_stubbed_retriever(
        query_response={
            "documents": [["rendered content"]],
            "metadatas": [[chunk.to_metadata()]],
            "distances": [[0.42]],
        }
    )

    [result] = retriever.query("guide", top_k=1)

    assert result.chunk == DocumentChunk(
        doc_id="doc-9",
        path=Path("docs/guide.md"),
        title="Guide",
        filetype="markdown",
        chunk_index=4,
        line_start=12,
        line_end=18,
        content="rendered content",
    )
    assert result.score == 0.42


def test_reset_ignores_missing_collection_delete_failure():
    retriever, collection, client = make_stubbed_retriever(
        delete_exception=chromadb.errors.NotFoundError("missing")
    )

    retriever.reset()

    assert client.deleted_names == ["test"]
    assert client.created_names == ["test"]
    assert retriever.collection is collection


def test_reset_reraises_unexpected_delete_failures():
    retriever, _, client = make_stubbed_retriever(delete_exception=RuntimeError("disk full"))

    with pytest.raises(RuntimeError, match="disk full"):
        retriever.reset()

    assert client.created_names == []
