from __future__ import annotations

import json
from pathlib import Path

from agenticrag.models import DocumentChunk, RetrievedChunk
from agenticrag.state import ConversationState
from agenticrag.tools import TOOL_SCHEMAS, RetrievalTools


class StubRetriever:
    def __init__(self) -> None:
        self.chunk = DocumentChunk(
            doc_id="doc-1",
            path=Path("docs/guide.md"),
            title="Guide",
            filetype="md",
            chunk_index=0,
            line_start=1,
            line_end=2,
            content="Alpha retrieval content",
        )
        self.calls: list[tuple[str, int]] = []

    def query(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return [RetrievedChunk(chunk=self.chunk, score=0.12)]


def write_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_id": "doc-1",
        "path": "docs/guide.md",
        "title": "Guide",
        "filetype": "md",
        "lines": [
            "Intro line",
            "Alpha retrieval content",
            "Beta follow up",
            "alpha second mention",
        ],
    }
    (cache_dir / "doc-1.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_search_assigns_reference_and_stores_tool_result(tmp_path):
    state = ConversationState(user_query="what is alpha?")
    retriever = StubRetriever()
    tools = RetrievalTools(retriever=retriever, state=state, source_cache_dir=tmp_path)

    result = tools.search(["alpha", "alpha"])

    assert "[turn0search0]" in result
    assert "Alpha retrieval content" in result
    assert list(state.references) == ["turn0search0"]
    assert state.references["turn0search0"].chunk == retriever.chunk
    assert state.tool_results[-1].name == "search"
    assert state.tool_results[-1].content == result
    assert state.tool_results[-1].metadata == {"reference_ids": ["turn0search0"]}
    assert retriever.calls == [("alpha", 10), ("alpha", 10)]


def test_find_and_open_use_source_cache(tmp_path):
    write_cache(tmp_path)
    state = ConversationState(user_query="what is alpha?")
    tools = RetrievalTools(retriever=StubRetriever(), state=state, source_cache_dir=tmp_path)
    tools.search(["alpha"])

    find_result = tools.find("turn0search0", ["alpha"])
    open_result = tools.open("turn0search0", line_number=1)
    default_open_result = tools.open("turn0search0")

    assert "line 2" in find_result
    assert "Alpha retrieval content" in find_result
    assert "line 4" in find_result
    assert "alpha second mention" in find_result
    assert "Viewing lines [1-4] of 4 total lines" in open_result
    assert "1: Intro line" in open_result
    assert "2: Alpha retrieval content" in open_result
    assert state.tool_results[-3].name == "find"
    assert state.tool_results[-3].metadata == {"reference_ids": ["turn0search0"]}
    assert state.tool_results[-2].name == "open"
    assert state.tool_results[-2].metadata == {"reference_ids": ["turn0search0"]}
    assert state.tool_results[-1].name == "open"
    assert state.tool_results[-1].metadata == {"reference_ids": ["turn0search0"]}
    assert default_open_result.startswith("Viewing lines [1-4]")


def test_tool_schemas_include_exact_tool_names():
    assert [schema["function"]["name"] for schema in TOOL_SCHEMAS] == [
        "search",
        "find",
        "open",
        "summarize",
    ]
