from pathlib import Path

from agenticrag.models import DocumentChunk, RetrievedChunk


def test_document_chunk_to_metadata_preserves_fields():
    chunk = DocumentChunk(
        doc_id="doc-1",
        path=Path("docs") / "guide.md",
        title="Guide",
        filetype="markdown",
        chunk_index=3,
        line_start=10,
        line_end=24,
        content="A useful chunk",
    )

    assert chunk.to_metadata() == {
        "doc_id": "doc-1",
        "path": "docs/guide.md",
        "title": "Guide",
        "filetype": "markdown",
        "chunk_index": 3,
        "line_start": 10,
        "line_end": 24,
    }


def test_retrieved_chunk_snippet_returns_first_200_chars():
    content = "x" * 250
    chunk = DocumentChunk(
        doc_id="doc-1",
        path=Path("docs/guide.md"),
        title="Guide",
        filetype="markdown",
        chunk_index=0,
        line_start=1,
        line_end=20,
        content=content,
    )
    retrieved = RetrievedChunk(chunk=chunk, score=0.92)

    assert retrieved.snippet == "x" * 200
