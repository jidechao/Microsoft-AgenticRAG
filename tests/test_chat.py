from pathlib import Path

from agenticrag.chat import build_rewrite_messages, parse_rewrite_response, rewrite_query
from agenticrag.models import DocumentChunk, Reference
from agenticrag.prompts import QUERY_REWRITE_PROMPT
from agenticrag.state import ConversationState


def test_parse_rewrite_response_extracts_query():
    assert (
        parse_rewrite_response('{"query": "请详细说明第二个模块"}', "fallback")
        == "请详细说明第二个模块"
    )


def test_parse_rewrite_response_falls_back_for_invalid_json_empty_and_non_string_query():
    assert parse_rewrite_response("not json", "原问题") == "原问题"
    assert parse_rewrite_response('{"query": "   "}', "原问题") == "原问题"
    assert parse_rewrite_response('{"query": 123}', "原问题") == "原问题"


def test_build_rewrite_messages_includes_prompt_history_question_and_references():
    state = ConversationState(user_query="什么是AgenticRAG？")
    state.add_message("assistant", "AgenticRAG是一种检索增强生成方法。")
    state.add_tool_result("search", "tool output should be skipped")
    state.add_message("user", "第二个模块是什么？")
    state.references["turn0search0"] = Reference(
        reference_id="turn0search0",
        chunk=DocumentChunk(
            doc_id="doc-1",
            path=Path("docs/design.md"),
            title="AgenticRAG设计文档",
            filetype="markdown",
            chunk_index=0,
            line_start=0,
            line_end=10,
            content="第二个模块介绍",
        ),
    )

    messages = build_rewrite_messages(state, "它有什么作用？")

    assert messages[0] == {"role": "system", "content": QUERY_REWRITE_PROMPT}
    user_message = messages[1]["content"]
    assert "Recent conversation:" in user_message
    assert "user: 什么是AgenticRAG？" in user_message
    assert "assistant: AgenticRAG是一种检索增强生成方法。" in user_message
    assert "tool output should be skipped" not in user_message
    assert "Current user question:" in user_message
    assert "它有什么作用？" in user_message
    assert "turn0search0" in user_message
    assert "AgenticRAG设计文档" in user_message
    assert "docs/design.md" in user_message


def test_build_rewrite_messages_omits_empty_summary_placeholders():
    state = ConversationState(user_query="initial question")
    state.messages.clear()

    messages = build_rewrite_messages(state, "current question")
    user_message = messages[1]["content"]

    assert "Recent conversation:" in user_message
    assert "Available Reference IDs:" in user_message
    assert "Current user question:" in user_message
    assert "current question" in user_message
    assert "No prior conversation." not in user_message
    assert "No Reference IDs yet." not in user_message


def test_rewrite_query_calls_complete_with_messages_keyword_and_returns_parsed_query():
    class FakeLLMClient:
        def __init__(self):
            self.messages = None

        def complete(self, *, messages):
            self.messages = messages
            return '{"query": "请详细说明AgenticRAG的第二个模块"}'

    state = ConversationState(user_query="第二个模块是什么？")
    llm_client = FakeLLMClient()

    assert rewrite_query(llm_client, state, "它有什么作用？") == "请详细说明AgenticRAG的第二个模块"
    assert llm_client.messages == build_rewrite_messages(state, "它有什么作用？")


def test_rewrite_query_falls_back_for_empty_json():
    class FakeLLMClient:
        def complete(self, *, messages):
            return "{}"

    state = ConversationState(user_query="第二个模块是什么？")

    assert rewrite_query(FakeLLMClient(), state, "它有什么作用？") == "它有什么作用？"
