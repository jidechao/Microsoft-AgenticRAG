from __future__ import annotations

import json
from pathlib import Path

from agenticrag.models import DocumentChunk, RetrievedChunk
from agenticrag.state import ConversationState
from agenticrag.tools import TOOL_SCHEMAS, RetrievalTools


def make_chunk(
    *,
    doc_id: str = "doc-1",
    path: str = "docs/guide.md",
    title: str = "Guide",
    chunk_index: int = 0,
    line_start: int = 1,
    line_end: int = 1,
    content: str = "Alpha retrieval content",
) -> DocumentChunk:
    return DocumentChunk(
        doc_id=doc_id,
        path=Path(path),
        title=title,
        filetype="md",
        chunk_index=chunk_index,
        line_start=line_start,
        line_end=line_end,
        content=content,
    )


class StubRetriever:
    def __init__(self, results_by_query: dict[str, list[DocumentChunk]] | None = None) -> None:
        self.default_chunk = make_chunk()
        self.results_by_query = results_by_query or {"alpha": [self.default_chunk]}
        self.calls: list[tuple[str, int]] = []

    def query(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        chunks = self.results_by_query.get(query, self.results_by_query.get("*", []))
        return [RetrievedChunk(chunk=chunk, score=0.12) for chunk in chunks]


def write_cache(
    cache_dir: Path,
    *,
    doc_id: str = "doc-1",
    path: str = "docs/guide.md",
    lines: list[str] | None = None,
    extra: dict | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_id": doc_id,
        "path": path,
        "title": "Guide",
        "filetype": "md",
        "lines": lines
        or [
            "Intro line",
            "Alpha retrieval content",
            "Beta follow up",
            "alpha second mention",
        ],
    }
    if extra:
        payload.update(extra)
    (cache_dir / f"{doc_id}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def make_tools(tmp_path: Path, retriever: StubRetriever | None = None) -> tuple[RetrievalTools, ConversationState, StubRetriever]:
    state = ConversationState(user_query="what is alpha?")
    stub = retriever or StubRetriever()
    tools = RetrievalTools(retriever=stub, state=state, source_cache_dir=tmp_path)
    return tools, state, stub


def test_search_assigns_reference_and_stores_tool_result(tmp_path):
    tools, state, retriever = make_tools(tmp_path)

    result = tools.search([" alpha ", "", "alpha"])

    assert "[turn0search0]" in result
    assert "docs/guide.md:2-2" in result
    assert "Alpha retrieval content" in result
    assert list(state.references) == ["turn0search0"]
    assert state.references["turn0search0"].chunk == retriever.default_chunk
    assert state.tool_results[-1].name == "search"
    assert state.tool_results[-1].content == result
    assert state.tool_results[-1].metadata == {"reference_ids": ["turn0search0"]}
    assert retriever.calls == [("alpha", 10)]


def test_find_and_open_use_source_cache_with_one_based_display(tmp_path):
    write_cache(tmp_path)
    tools, state, _ = make_tools(tmp_path)
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
    assert default_open_result.startswith("Viewing lines [1-4]")
    assert state.tool_results[-3].name == "find"
    assert state.tool_results[-3].metadata == {"reference_ids": ["turn0search0"]}
    assert state.tool_results[-2].name == "open"
    assert state.tool_results[-2].metadata == {"reference_ids": ["turn0search0"]}
    assert state.tool_results[-1].name == "open"
    assert state.tool_results[-1].metadata == {"reference_ids": ["turn0search0"]}


def test_default_open_uses_zero_based_chunk_line_start(tmp_path):
    chunk = make_chunk(line_start=2, line_end=2, content="Target")
    retriever = StubRetriever({"target": [chunk]})
    write_cache(tmp_path, lines=["L1", "L2", "Target", "L4"])
    tools, _, _ = make_tools(tmp_path, retriever)
    tools.search(["target"])

    result = tools.open("turn0search0")

    assert "Viewing lines [1-4] of 4 total lines" in result
    assert "3: Target" in result


def test_find_and_open_return_tool_errors_for_mismatched_cache(tmp_path):
    write_cache(tmp_path, path="docs/stale.md")
    tools, state, _ = make_tools(tmp_path)
    tools.search(["alpha"])

    find_result = tools.find("turn0search0", ["alpha"])
    open_result = tools.open("turn0search0")

    assert find_result.startswith("[tool error] find:")
    assert "cached path" in find_result
    assert open_result.startswith("[tool error] open:")
    assert "cached path" in open_result
    assert state.tool_results[-2].name == "find"
    assert state.tool_results[-1].name == "open"


def test_invalid_cache_shape_returns_tool_error(tmp_path):
    write_cache(tmp_path, extra={"lines": "not-a-list"})
    tools, state, _ = make_tools(tmp_path)
    tools.search(["alpha"])

    result = tools.open("turn0search0")

    assert result.startswith("[tool error] open:")
    assert "invalid lines" in result
    assert state.tool_results[-1].content == result


def test_unknown_reference_ids_do_not_crash_find_open_or_summarize(tmp_path):
    tools, state, _ = make_tools(tmp_path)

    find_result = tools.find("missing", ["alpha"])
    open_result = tools.open("missing")
    summarize_result = tools.summarize(["missing"])

    assert find_result == "[tool error] find: Unknown reference_id: missing"
    assert open_result == "[tool error] open: Unknown reference_id: missing"
    assert summarize_result == "[tool error] summarize: unknown reference_id(s): missing"
    assert [tool.name for tool in state.tool_results] == ["find", "open", "summarize"]
    assert state.tool_results[-1].metadata == {
        "reference_ids": [],
        "unknown_reference_ids": ["missing"],
    }
    assert [message["content"] for message in state.messages if message["role"] == "tool"] == [
        find_result,
        open_result,
        summarize_result,
    ]


def test_search_preserves_duplicate_content_with_different_locations(tmp_path):
    first = make_chunk(chunk_index=0, line_start=0, line_end=0, content="same")
    second = make_chunk(chunk_index=1, line_start=4, line_end=4, content="same")
    retriever = StubRetriever({"same": [first, second]})
    tools, state, _ = make_tools(tmp_path, retriever)

    result = tools.search(["same"])

    assert "[turn0search0]" in result
    assert "[turn0search1]" in result
    assert "docs/guide.md:1-1" in result
    assert "docs/guide.md:5-5" in result
    assert list(state.references) == ["turn0search0", "turn0search1"]


def test_search_caps_queries_and_results(tmp_path):
    chunks = [
        make_chunk(chunk_index=index, line_start=index, line_end=index, content=f"chunk {index}")
        for index in range(12)
    ]
    retriever = StubRetriever({"*": chunks})
    tools, state, _ = make_tools(tmp_path, retriever)

    result = tools.search([" q0 ", "q1", "q2", "q3", "q4", "q5"])

    assert len(retriever.calls) == 1
    assert retriever.calls == [("q0", 10)]
    assert len(state.references) == 10
    assert "[turn0search9]" in result
    assert "chunk 10" not in result


def test_find_empty_patterns_returns_visible_result(tmp_path):
    write_cache(tmp_path)
    tools, state, _ = make_tools(tmp_path)
    tools.search(["alpha"])

    result = tools.find("turn0search0", [" ", ""])

    assert result == "No non-empty patterns provided."
    assert state.tool_results[-1].name == "find"
    assert state.tool_results[-1].metadata == {"reference_ids": ["turn0search0"]}


def test_summarize_with_mixed_refs_preserves_valid_refs_and_messages(tmp_path):
    first = make_chunk(doc_id="doc-1", content="first")
    second = make_chunk(doc_id="doc-2", path="docs/second.md", content="second")
    retriever = StubRetriever({"alpha": [first, second]})
    tools, state, _ = make_tools(tmp_path, retriever)
    tools.search(["alpha"])
    state.add_tool_result("find", "details for first", metadata={"reference_ids": ["turn0search0"]})
    state.add_tool_result("open", "details for second", metadata={"reference_ids": ["turn0search1"]})

    result = tools.summarize(["missing", "turn0search0", "missing"])

    assert result == "Summarized prior tool results. Ignored unknown reference_id(s): missing"
    assert state.tool_results[-1].name == "summarize"
    assert state.tool_results[-1].metadata == {
        "reference_ids": ["turn0search0"],
        "unknown_reference_ids": ["missing"],
    }
    assert state.tool_results[1].content == "details for first"
    assert state.tool_results[2].content == "[compressed open result unrelated to retained references]"
    tool_messages = [message["content"] for message in state.messages if message["role"] == "tool"]
    assert tool_messages[-1] == result
    assert "details for first" in tool_messages
    assert "[compressed open result unrelated to retained references]" in tool_messages


def test_tool_schemas_include_exact_tool_names_and_tighter_constraints():
    assert [schema["function"]["name"] for schema in TOOL_SCHEMAS] == [
        "search",
        "find",
        "open",
        "summarize",
    ]
    search_queries = TOOL_SCHEMAS[0]["function"]["parameters"]["properties"]["queries"]
    find_patterns = TOOL_SCHEMAS[1]["function"]["parameters"]["properties"]["patterns"]
    summarize_refs = TOOL_SCHEMAS[3]["function"]["parameters"]["properties"]["candidate_reference_ids"]
    assert search_queries["items"]["minLength"] == 1
    assert find_patterns["maxItems"] == 10
    assert find_patterns["items"]["minLength"] == 1
    assert summarize_refs["maxItems"] == 20
    assert summarize_refs["items"]["minLength"] == 1
