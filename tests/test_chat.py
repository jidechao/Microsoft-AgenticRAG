from pathlib import Path

from agenticrag.chat import (
    ChatSession,
    build_rewrite_messages,
    parse_rewrite_response,
    rewrite_query,
)
from agenticrag.models import DocumentChunk, Reference
from agenticrag.prompts import CHAT_SIMPLE_RAG_PROMPT, QUERY_REWRITE_PROMPT
from agenticrag.prompts import SYSTEM_PROMPT
from agenticrag.state import ConversationState


def test_run_chat_handles_help_reset_and_exit(monkeypatch, capsys):
    from agenticrag import chat

    class FakeSession:
        def __init__(self):
            self.reset_count = 0

        def answer_turn(self, user_input, status_writer=None):
            yield f"answer:{user_input}"

        def reset(self):
            self.reset_count += 1

    fake_session = FakeSession()
    inputs = iter(["/help", "", "hello", "/reset", "/exit"])

    monkeypatch.setattr(chat, "create_chat_session", lambda: fake_session)
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    assert chat.run_chat() == 0

    captured = capsys.readouterr()
    assert "AgenticRAG chat" in captured.out
    assert "/reset" in captured.out
    assert "answer:hello" in captured.out
    assert "Session reset." in captured.out
    assert fake_session.reset_count == 1


def test_parse_rewrite_response_extracts_fenced_json():
    response = '```json\n{"query": "explain module two"}\n```'

    assert parse_rewrite_response(response, "fallback") == "explain module two"


def test_parse_rewrite_response_extracts_plain_fenced_json():
    response = '```\n{"query": "explain module two"}\n```'

    assert parse_rewrite_response(response, "fallback") == "explain module two"


def test_parse_rewrite_response_extracts_json_with_surrounding_text():
    response = 'Here is the JSON: {"query": "explain module two"}'

    assert parse_rewrite_response(response, "fallback") == "explain module two"


def test_rewrite_query_falls_back_when_complete_raises():
    class FakeLLMClient:
        def complete(self, *, messages):
            raise RuntimeError("llm unavailable")

    state = ConversationState(user_query="initial question")

    assert rewrite_query(FakeLLMClient(), state, "original user input") == "original user input"


def test_chat_simple_rag_prompt_covers_required_behavior():
    assert "原始用户问题" in CHAT_SIMPLE_RAG_PROMPT
    assert "改写后的自包含问题" in CHAT_SIMPLE_RAG_PROMPT
    assert "检索结果" in CHAT_SIMPLE_RAG_PROMPT
    assert "Reference ID" in CHAT_SIMPLE_RAG_PROMPT
    assert "引用" in CHAT_SIMPLE_RAG_PROMPT
    assert "证据不足" in CHAT_SIMPLE_RAG_PROMPT


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


def test_chat_session_simple_route_rewrites_searches_streams_and_records_state(monkeypatch):
    raw_query = "它有什么作用？"
    rewritten_query = "请详细说明AgenticRAG的第二个模块有什么作用？"

    class FakeLLMClient:
        def stream(self, messages):
            raise AssertionError("stream_simple_chat is monkeypatched")

    class FakeTools:
        def __init__(self):
            self.search_calls = []
            self.state = None

        def search(self, queries):
            self.search_calls.append(queries)
            self.state.references["turn0search0"] = Reference(
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
            return "[turn0search0] context"

    def fake_rewrite(llm_client, state, user_input):
        assert user_input == raw_query
        assert state.messages[-1] == {"role": "user", "content": raw_query}
        return rewritten_query

    def fake_classify(llm_client, query):
        assert query == rewritten_query
        return "simple"

    def fake_stream_simple(llm_client, raw_value, rewritten_value, search_context):
        assert raw_value == raw_query
        assert rewritten_value == rewritten_query
        assert search_context == "[turn0search0] context"
        yield "chunk one "
        yield "chunk two"

    monkeypatch.setattr("agenticrag.chat.rewrite_query", fake_rewrite)
    monkeypatch.setattr("agenticrag.chat.classify_query", fake_classify)
    monkeypatch.setattr("agenticrag.chat.stream_simple_chat", fake_stream_simple)

    tools = FakeTools()
    session = ChatSession(
        llm_client=FakeLLMClient(),
        tools=tools,
        max_calls=3,
        token_threshold=10000,
        token_warning_ratio=0.8,
    )
    tools.state = session.state

    chunks = list(session.answer_turn(raw_query))

    assert chunks == ["chunk one ", "chunk two"]
    assert tools.search_calls == [[rewritten_query]]
    assert session.state.messages == [
        {"role": "user", "content": raw_query},
        {"role": "assistant", "content": "chunk one chunk two"},
    ]
    assert "turn0search0" in session.state.references


def test_chat_session_complex_route_streams_status_and_records_answer(monkeypatch):
    raw_query = "What does the second module do?"
    rewritten_query = "Explain what the second AgenticRAG module does."
    status_updates = []
    capture_status_writer = status_updates.append

    class FakeLLMClient:
        pass

    class FakeTools:
        def __init__(self):
            self.search_calls = []
            self.state = None

        def search(self, queries):
            self.search_calls.append(queries)
            self.state.references["turn0search0"] = Reference(
                reference_id="turn0search0",
                chunk=DocumentChunk(
                    doc_id="doc-1",
                    path=Path("docs/design.md"),
                    title="AgenticRAG design",
                    filetype="markdown",
                    chunk_index=0,
                    line_start=0,
                    line_end=10,
                    content="The second module coordinates retrieval tools.",
                ),
            )
            return "[turn0search0] context"

    def fake_rewrite(llm_client, state, user_input):
        assert user_input == raw_query
        assert state.messages[-1] == {"role": "user", "content": raw_query}
        return rewritten_query

    def fake_classify(llm_client, query):
        assert query == rewritten_query
        return "complex"

    def fake_run_agentic_loop(
        llm_client,
        state,
        tool_executor,
        max_calls,
        token_threshold,
        token_warning_ratio,
        status_writer,
    ):
        assert llm_client is fake_llm_client
        assert state is session.state
        assert max_calls == 3
        assert token_threshold == 10000
        assert token_warning_ratio == 0.8
        assert status_writer is capture_status_writer
        latest_message = state.messages[-1]
        assert latest_message["role"] == "user"
        assert raw_query in latest_message["content"]
        assert rewritten_query in latest_message["content"]
        status_writer("[tool] search")
        assert tool_executor("search", {"queries": [rewritten_query]}) == "[turn0search0] context"
        yield "complex chunk one "
        yield "complex chunk two"

    monkeypatch.setattr("agenticrag.chat.rewrite_query", fake_rewrite)
    monkeypatch.setattr("agenticrag.chat.classify_query", fake_classify)
    monkeypatch.setattr("agenticrag.chat.run_agentic_loop", fake_run_agentic_loop)

    fake_llm_client = FakeLLMClient()
    tools = FakeTools()
    session = ChatSession(
        llm_client=fake_llm_client,
        tools=tools,
        max_calls=3,
        token_threshold=10000,
        token_warning_ratio=0.8,
    )

    chunks = list(session.answer_turn(raw_query, status_writer=capture_status_writer))

    assert chunks == ["complex chunk one ", "complex chunk two"]
    assert status_updates == ["[tool] search"]
    assert tools.search_calls == [[rewritten_query]]
    user_messages = [
        message for message in session.state.messages if message["role"] == "user"
    ]
    assert len(user_messages) == 1
    assert raw_query in user_messages[0]["content"]
    assert rewritten_query in user_messages[0]["content"]
    assert session.state.messages[-1] == {
        "role": "assistant",
        "content": "complex chunk one complex chunk two",
    }
    assert "turn0search0" in session.state.references


def test_chat_session_complex_route_removes_internal_loop_messages_and_keeps_refs(monkeypatch):
    raw_query = "Where is retrieval coordinated?"
    rewritten_query = "Explain where AgenticRAG coordinates retrieval."

    class FakeLLMClient:
        pass

    class FakeTools:
        def __init__(self):
            self.state = None

        def search(self, queries):
            self.state.references["turn0search0"] = Reference(
                reference_id="turn0search0",
                chunk=DocumentChunk(
                    doc_id="doc-1",
                    path=Path("docs/design.md"),
                    title="AgenticRAG design",
                    filetype="markdown",
                    chunk_index=0,
                    line_start=0,
                    line_end=10,
                    content="Retrieval is coordinated by the agentic loop.",
                ),
            )
            self.state.add_tool_result(
                "search",
                "[turn0search0] context",
                metadata={"reference_ids": ["turn0search0"]},
            )
            return "[turn0search0] context"

    def fake_run_agentic_loop(
        llm_client,
        state,
        tool_executor,
        max_calls,
        token_threshold,
        token_warning_ratio,
        status_writer,
    ):
        state.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        state.add_message(
            "assistant",
            "",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        )
        tool_executor("search", {"queries": [rewritten_query]})
        yield "final answer"

    monkeypatch.setattr(
        "agenticrag.chat.rewrite_query",
        lambda llm_client, state, user_input: rewritten_query,
    )
    monkeypatch.setattr(
        "agenticrag.chat.classify_query",
        lambda llm_client, query: "complex",
    )
    monkeypatch.setattr("agenticrag.chat.run_agentic_loop", fake_run_agentic_loop)

    tools = FakeTools()
    session = ChatSession(
        llm_client=FakeLLMClient(),
        tools=tools,
        max_calls=3,
        token_threshold=10000,
        token_warning_ratio=0.8,
    )

    assert list(session.answer_turn(raw_query)) == ["final answer"]

    assert session.state.messages == [
        {
            "role": "user",
            "content": "\n".join(
                [
                    f"Raw user question: {raw_query}",
                    f"Rewritten self-contained question: {rewritten_query}",
                ]
            ),
        },
        {"role": "assistant", "content": "final answer"},
    ]
    assert all(message.get("content") != SYSTEM_PROMPT for message in session.state.messages)
    assert not any(
        message.get("role") == "assistant"
        and message.get("content") == ""
        and "tool_calls" in message
        for message in session.state.messages
    )
    assert "turn0search0" in session.state.references
    assert [tool_result.name for tool_result in session.state.tool_results] == ["search"]


def test_chat_session_reset_clears_state_refs_and_keeps_empty_fresh_history():
    class FakeLLMClient:
        pass

    class FakeTools:
        def __init__(self):
            self.state = None

    tools = FakeTools()
    session = ChatSession(
        llm_client=FakeLLMClient(),
        tools=tools,
        max_calls=3,
        token_threshold=10000,
        token_warning_ratio=0.8,
    )
    session.state.add_message("user", "previous question")
    session.state.references["turn0search0"] = Reference(
        reference_id="turn0search0",
        chunk=DocumentChunk(
            doc_id="doc-1",
            path=Path("docs/design.md"),
            title="AgenticRAG design",
            filetype="markdown",
            chunk_index=0,
            line_start=0,
            line_end=10,
            content="Previous context.",
        ),
    )

    session.reset()

    assert session.state.references == {}
    assert tools.state is session.state
    # ChatSession deliberately hides ConversationState's synthetic empty initial user message.
    assert session.state.messages == []
