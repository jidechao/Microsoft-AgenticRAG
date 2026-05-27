from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agenticrag.models import DocumentChunk, Reference, RetrievedChunk, ToolCall, ToolResult


def make_document_chunk(content: str = "A useful chunk") -> DocumentChunk:
    return DocumentChunk(
        doc_id="doc-1",
        path=Path("docs") / "guide.md",
        title="Guide",
        filetype="markdown",
        chunk_index=3,
        line_start=10,
        line_end=24,
        content=content,
    )


def test_document_chunk_to_metadata_preserves_fields():
    chunk = make_document_chunk()

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
    chunk = make_document_chunk(content="x" * 250)
    retrieved = RetrievedChunk(chunk=chunk, score=0.92)

    assert retrieved.snippet == "x" * 200


def test_tool_result_metadata_uses_isolated_default_dicts():
    first = ToolResult(name="search", content="first")
    second = ToolResult(name="search", content="second")

    first.metadata["source"] = "docs"

    assert second.metadata == {}


def test_reference_is_immutable():
    reference = Reference(reference_id="ref-1", chunk=make_document_chunk())

    with pytest.raises(FrozenInstanceError):
        reference.reference_id = "ref-2"


def test_tool_call_arguments_are_immutable_and_copied():
    arguments = {"query": "agentic rag"}
    tool_call = ToolCall(id="call-1", name="search", arguments=arguments)

    with pytest.raises(TypeError):
        tool_call.arguments["query"] = "changed"

    arguments["query"] = "mutated outside"

    assert tool_call.arguments["query"] == "agentic rag"
