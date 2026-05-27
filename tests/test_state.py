from pathlib import Path

from agenticrag.models import DocumentChunk
from agenticrag.state import ConversationState


def make_chunk(content: str, chunk_index: int) -> DocumentChunk:
    return DocumentChunk(
        doc_id=f"doc-{chunk_index}",
        path=Path("docs") / f"doc-{chunk_index}.md",
        title=f"Doc {chunk_index}",
        filetype="markdown",
        chunk_index=chunk_index,
        line_start=1,
        line_end=3,
        content=content,
    )


def test_assign_search_results_creates_turn_scoped_reference_ids():
    state = ConversationState(user_query="问题")

    reference_ids = state.assign_search_results(
        [make_chunk("alpha", 0), make_chunk("beta", 1)]
    )

    assert reference_ids == ["turn0search0", "turn0search1"]
    assert state.get_reference("turn0search1").chunk.content == "beta"


def test_summarize_preserves_retained_tool_results_and_reference_mapping():
    state = ConversationState(user_query="问题")
    retained_id, unrelated_id = state.assign_search_results(
        [make_chunk("alpha", 0), make_chunk("beta", 1)]
    )
    state.add_tool_result(
        "search",
        "retained full result content",
        metadata={"reference_ids": [retained_id]},
    )
    state.add_tool_result(
        "search",
        "unrelated full result content",
        metadata={"reference_ids": [unrelated_id]},
    )

    state.summarize([retained_id])

    assert state.tool_results[0].content == "retained full result content"
    assert state.tool_results[1].content != "unrelated full result content"
    assert "compressed" in state.tool_results[1].content.lower()
    assert state.get_reference(retained_id).chunk.content == "alpha"

    tool_messages = [message for message in state.messages if message["role"] == "tool"]
    assert [message["content"] for message in tool_messages] == [
        state.tool_results[0].content,
        state.tool_results[1].content,
    ]
