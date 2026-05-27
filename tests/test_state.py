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


def test_repeated_assign_search_results_produces_distinct_ids():
    state = ConversationState(user_query="问题")

    first = state.assign_search_results([make_chunk("alpha", 0)])
    second = state.assign_search_results([make_chunk("beta", 1)])

    assert first == ["turn0search0"]
    assert second == ["turn1search0"]
    assert state.get_reference("turn0search0").chunk.content == "alpha"
    assert state.get_reference("turn1search0").chunk.content == "beta"


def test_maybe_add_token_warning_only_adds_once():
    state = ConversationState(user_query="问题")
    state.add_message("assistant", "x" * 50)

    threshold = state.total_tokens() + 1
    first = state.maybe_add_token_warning(threshold=threshold, ratio=0.5)
    second = state.maybe_add_token_warning(threshold=threshold, ratio=0.5)

    assert first is True
    assert second is False
    system_messages = [
        message for message in state.messages if message["role"] == "system"
    ]
    assert len(system_messages) == 1


def test_summarize_compresses_missing_reference_ids_and_preserves_order():
    state = ConversationState(user_query="问题")
    retained_id, missing_id = state.assign_search_results(
        [make_chunk("alpha", 0), make_chunk("beta", 1)]
    )
    state.add_message("assistant", "before tool")
    state.add_tool_result(
        "search",
        "retained full result content",
        metadata={"reference_ids": [retained_id]},
    )
    state.add_message("assistant", "middle")
    state.add_tool_result(
        "search",
        "missing reference result content",
        metadata={"reference_ids": []},
    )
    state.add_message("assistant", "after tool")

    before_roles = [message["role"] for message in state.messages]
    before_tool_contents = [
        message["content"] for message in state.messages if message["role"] == "tool"
    ]

    state.summarize([retained_id])

    after_roles = [message["role"] for message in state.messages]
    assert after_roles == before_roles
    assert [message["content"] for message in state.messages if message["role"] == "tool"] == [
        "retained full result content",
        "[compressed search result unrelated to retained references]",
    ]
    assert state.tool_results[1].content == "[compressed search result unrelated to retained references]"
    assert before_tool_contents[0] == "retained full result content"


def test_repeated_summarize_is_stable_for_tested_cases():
    state = ConversationState(user_query="问题")
    retained_id, _ = state.assign_search_results([make_chunk("alpha", 0), make_chunk("beta", 1)])
    state.add_tool_result(
        "search",
        "retained full result content",
        metadata={"reference_ids": [retained_id]},
    )
    state.add_tool_result("search", "missing reference result content", metadata={})

    state.summarize([retained_id])
    first_snapshot = [message.copy() for message in state.messages]
    first_tool_contents = [tool.content for tool in state.tool_results]

    state.summarize([retained_id])

    assert state.messages == first_snapshot
    assert [tool.content for tool in state.tool_results] == first_tool_contents
