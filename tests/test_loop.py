from pathlib import Path
from types import SimpleNamespace

from agenticrag.models import DocumentChunk
from agenticrag.models import Reference
from agenticrag.loop import execute_retrieval_tool
from agenticrag.loop import CURRENT_TURN_TOOL_REQUIRED_PROMPT
from agenticrag.loop import run_agentic_loop
from agenticrag.loop import should_force_completion
from agenticrag.loop import stream_simple_rag
from agenticrag.prompts import FORCE_FINAL_ANSWER_PROMPT
from agenticrag.prompts import SIMPLE_RAG_PROMPT
from agenticrag.prompts import SYSTEM_PROMPT
from agenticrag.state import ConversationState
from agenticrag.tools import TOOL_SCHEMAS

QUESTION_LABEL = "\u95ee\u9898\uff1a"
SEARCH_RESULTS_LABEL = "\u68c0\u7d22\u7ed3\u679c\uff1a"


class FakeLLM:
    def __init__(self, responses=None, stream_chunks=None):
        self.responses = list(responses or [])
        self.stream_chunks = list(stream_chunks or [])
        self.tool_call_calls = []
        self.stream_calls = []

    def tool_call(self, messages, tools):
        self.tool_call_calls.append((messages, tools))
        return self.responses.pop(0)

    def stream(self, messages):
        self.stream_calls.append(messages)
        yield from self.stream_chunks


class FakeRetrievalTools:
    def __init__(self):
        self.calls = []

    def search(self, queries):
        self.calls.append(("search", queries))
        return "search result"

    def find(self, reference_id, patterns):
        self.calls.append(("find", reference_id, patterns))
        return "find result"

    def open(self, reference_id, *, line_number):
        self.calls.append(("open", reference_id, line_number))
        return "open result"

    def summarize(self, candidate_reference_ids):
        self.calls.append(("summarize", candidate_reference_ids))
        return "summary result"


def message_response(content="", tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                ),
            ),
        ],
    )


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def add_reference_result(
    state,
    name="search",
    reference_id="turn0search0",
    content="tool result",
):
    state.references[reference_id] = Reference(
        reference_id=reference_id,
        chunk=DocumentChunk(
            doc_id=f"doc-{reference_id}",
            path=Path(f"docs/{reference_id}.md"),
            title="Reference",
            filetype="markdown",
            chunk_index=0,
            line_start=0,
            line_end=1,
            content="Reference content.",
        ),
    )
    state.add_tool_result(
        name,
        content,
        metadata={"reference_ids": [reference_id]},
    )
    return content


def test_should_force_completion_at_max_calls():
    assert should_force_completion(call_index=15, max_calls=15) is True
    assert should_force_completion(call_index=14, max_calls=15) is False


def test_stream_simple_rag_uses_prompt_and_chinese_labels():
    llm = FakeLLM(stream_chunks=["answer"])

    assert list(stream_simple_rag(llm, query="what is A?", search_context="chunk 1")) == [
        "answer"
    ]

    messages = llm.stream_calls[0]
    assert messages[0] == {"role": "system", "content": SIMPLE_RAG_PROMPT}
    assert messages[1]["role"] == "user"
    assert QUESTION_LABEL in messages[1]["content"]
    assert SEARCH_RESULTS_LABEL in messages[1]["content"]
    assert "what is A?" in messages[1]["content"]
    assert "chunk 1" in messages[1]["content"]


def test_system_prompt_requires_current_turn_tool_call_for_complex_turns():
    assert "For every new complex user turn" in SYSTEM_PROMPT
    assert "call at least one retrieval tool" in SYSTEM_PROMPT
    assert "For comparison questions" in SYSTEM_PROMPT
    assert "current-turn tool call" in SYSTEM_PROMPT


def test_execute_retrieval_tool_dispatches_supported_tools():
    tools = FakeRetrievalTools()

    assert (
        execute_retrieval_tool(tools, "search", {"queries": ["q"]})
        == "search result"
    )
    assert execute_retrieval_tool(
        tools,
        "find",
        {"reference_id": "turn0search0", "patterns": ["agent"]},
    ) == "find result"
    assert execute_retrieval_tool(
        tools,
        "open",
        {"reference_id": "turn0search0", "line_number": 7},
    ) == "open result"
    assert execute_retrieval_tool(
        tools,
        "open",
        {"reference_id": "turn0search0"},
    ) == "open result"
    assert execute_retrieval_tool(
        tools,
        "summarize",
        {"candidate_reference_ids": ["turn0search0"]},
    ) == "summary result"

    assert tools.calls == [
        ("search", ["q"]),
        ("find", "turn0search0", ["agent"]),
        ("open", "turn0search0", 7),
        ("open", "turn0search0", 0),
        ("summarize", ["turn0search0"]),
    ]


def test_execute_retrieval_tool_returns_error_argument_message():
    tools = FakeRetrievalTools()

    assert execute_retrieval_tool(tools, "search", {"_error": "bad args"}) == (
        "[tool error] search: bad args"
    )
    assert tools.calls == []


def test_execute_retrieval_tool_returns_unknown_tool_error():
    tools = FakeRetrievalTools()

    assert execute_retrieval_tool(tools, "nope", {}) == (
        "[tool error] nope: unknown tool"
    )
    assert tools.calls == []


def test_execute_retrieval_tool_validates_argument_shapes():
    tools = FakeRetrievalTools()

    invalid_calls = [
        ("search", {"queries": "q"}),
        ("find", {"reference_id": 1, "patterns": ["agent"]}),
        ("find", {"reference_id": "turn0search0", "patterns": "agent"}),
        ("open", {"reference_id": 1, "line_number": 7}),
        ("open", {"reference_id": "turn0search0", "line_number": "7"}),
        ("summarize", {"candidate_reference_ids": "turn0search0"}),
    ]

    for name, arguments in invalid_calls:
        assert execute_retrieval_tool(tools, name, arguments).startswith(
            f"[tool error] {name}:"
        )
    assert tools.calls == []


def test_execute_retrieval_tool_catches_tool_exceptions():
    class FailingRetrievalTools(FakeRetrievalTools):
        def search(self, queries):
            raise RuntimeError("boom")

    assert execute_retrieval_tool(
        FailingRetrievalTools(),
        "search",
        {"queries": ["q"]},
    ) == "[tool error] search: boom"


def test_run_agentic_loop_executes_tool_and_returns_final_answer():
    tool = tool_call("call-1", "search", '{"queries": ["alpha"]}')
    llm = FakeLLM(
        responses=[
            message_response(tool_calls=[tool]),
            message_response(content="final answer"),
        ]
    )
    state = ConversationState(user_query="question")
    statuses = []
    tool_calls = []

    def execute(name, arguments):
        tool_calls.append((name, arguments))
        return add_reference_result(state, name)

    chunks = list(
        run_agentic_loop(
            llm,
            state,
            execute,
            max_calls=3,
            token_threshold=10_000,
            token_warning_ratio=0.8,
            status_writer=statuses.append,
            require_current_turn_retrieval=True,
        )
    )

    assert chunks == ["final answer"]
    assert llm.tool_call_calls[0][1] == TOOL_SCHEMAS
    assert statuses == ["[tool] search"]
    assert tool_calls == [("search", {"queries": ["alpha"]})]
    assert state.messages[-2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"queries": ["alpha"]}',
                },
            }
        ],
    }
    assert state.messages[-1] == {
        "role": "tool",
        "content": "tool result",
        "tool_call_id": "call-1",
        "name": "search",
    }


def test_run_agentic_loop_requires_current_turn_retrieval_before_final_answer():
    tool = tool_call("call-1", "search", '{"queries": ["alpha"]}')
    llm = FakeLLM(
        responses=[
            message_response(content="premature final from history"),
            message_response(tool_calls=[tool]),
            message_response(content="final answer"),
        ],
        stream_chunks=["final ", "answer"],
    )
    state = ConversationState(user_query="question")
    statuses = []
    tool_calls = []

    def execute(name, arguments):
        tool_calls.append((name, arguments))
        return add_reference_result(
            state,
            name,
            reference_id="turn0search0",
            content="current search result",
        )

    chunks = list(
        run_agentic_loop(
            llm,
            state,
            execute,
            max_calls=3,
            token_threshold=10_000,
            token_warning_ratio=0.8,
            status_writer=statuses.append,
            require_current_turn_retrieval=True,
        )
    )

    assert chunks == ["final ", "answer"]
    assert statuses == ["[tool] search", "[tool] search"]
    assert tool_calls == [
        ("search", {"queries": ["question"]}),
        ("search", {"queries": ["alpha"]}),
    ]
    assert not any(
        message.get("role") == "system"
        and message.get("content") == CURRENT_TURN_TOOL_REQUIRED_PROMPT
        for message in state.messages
    )


def test_run_agentic_loop_streams_tool_error_status():
    tool = tool_call("call-1", "search", '{"queries": "not-a-list"}')
    llm = FakeLLM(
        responses=[
            message_response(tool_calls=[tool]),
            message_response(content="final answer"),
        ]
    )
    state = ConversationState(user_query="question")
    statuses = []

    chunks = list(
        run_agentic_loop(
            llm,
            state,
            lambda name, arguments: execute_retrieval_tool(
                FakeRetrievalTools(),
                name,
                arguments,
            ),
            max_calls=2,
            token_threshold=10_000,
            token_warning_ratio=0.8,
            status_writer=statuses.append,
            require_current_turn_retrieval=True,
        )
    )

    assert chunks == ["证据不足：当前复杂问题没有成功的本轮检索结果，无法生成可靠回答。"]
    assert statuses == [
        "[tool] search",
        "[tool error] search: queries must be a list",
        "[tool] search",
    ]


def test_run_agentic_loop_streams_final_answer_after_tool_chain():
    search_tool = tool_call("call-1", "search", '{"queries": ["alpha"]}')
    find_tool = tool_call(
        "call-2",
        "find",
        '{"reference_id": "turn0search0", "patterns": ["agent"]}',
    )
    open_tool = tool_call(
        "call-3",
        "open",
        '{"reference_id": "turn0search0", "line_number": 3}',
    )
    llm = FakeLLM(
        responses=[
            message_response(tool_calls=[search_tool]),
            message_response(tool_calls=[find_tool]),
            message_response(tool_calls=[open_tool]),
            message_response(content="non-stream fallback answer"),
        ],
        stream_chunks=["stream ", "answer"],
    )
    state = ConversationState(user_query="question")
    statuses = []

    def execute(name, arguments):
        return add_reference_result(state, name, content=f"{name} result")

    chunks = list(
        run_agentic_loop(
            llm,
            state,
            execute,
            max_calls=4,
            token_threshold=10_000,
            token_warning_ratio=0.8,
            status_writer=statuses.append,
            require_current_turn_retrieval=True,
        )
    )

    assert chunks == ["stream ", "answer"]
    assert statuses == ["[tool] search", "[tool] find", "[tool] open"]
    assert len(llm.stream_calls) == 1
    stream_messages = llm.stream_calls[0]
    assert stream_messages is not state.messages
    assert not any(message.get("role") == "tool" for message in stream_messages)
    assert not any("tool_calls" in message for message in stream_messages)
    assert stream_messages[-1]["role"] == "system"
    assert "Collected tool evidence:" in stream_messages[-1]["content"]
    assert "[search]" in stream_messages[-1]["content"]
    assert "[find]" in stream_messages[-1]["content"]
    assert "[open]" in stream_messages[-1]["content"]


def test_run_agentic_loop_forces_final_answer_after_max_calls():
    tool = tool_call("call-1", "search", '{"queries": ["alpha"]}')
    llm = FakeLLM(
        responses=[message_response(tool_calls=[tool])],
        stream_chunks=["forced", " answer"],
    )
    state = ConversationState(user_query="question")
    def execute(name, arguments):
        return add_reference_result(state, name)

    chunks = list(
        run_agentic_loop(
            llm,
            state,
            execute,
            max_calls=1,
            token_threshold=10_000,
            token_warning_ratio=0.8,
            status_writer=lambda status: None,
        )
    )

    assert chunks == ["forced", " answer"]
    assert state.messages[-1] == {
        "role": "system",
        "content": FORCE_FINAL_ANSWER_PROMPT,
    }
    stream_messages = llm.stream_calls[0]
    assert stream_messages is not state.messages
    assert stream_messages[0]["role"] == "system"
    assert "Tool budget is exhausted." in stream_messages[0]["content"]
    assert not any(message.get("role") == "tool" for message in stream_messages)
    assert not any("tool_calls" in message for message in stream_messages)


def test_run_agentic_loop_summarizes_when_token_threshold_is_reached():
    llm = FakeLLM(responses=[message_response(content="done")])
    state = ConversationState(user_query="question")
    state.references["ref-1"] = SimpleNamespace()
    tool_calls = []

    list(
        run_agentic_loop(
            llm,
            state,
            lambda name, arguments: tool_calls.append((name, arguments)) or "summary",
            max_calls=1,
            token_threshold=0,
            token_warning_ratio=0.8,
            status_writer=lambda status: None,
        )
    )

    assert tool_calls == [("summarize", {"candidate_reference_ids": ["ref-1"]})]


def test_run_agentic_loop_inserts_system_prompt_once_at_beginning_across_runs():
    first_tool = tool_call("call-1", "search", '{"queries": ["first"]}')
    second_tool = tool_call("call-2", "search", '{"queries": ["second"]}')
    llm = FakeLLM(
        responses=[
            message_response(tool_calls=[first_tool]),
            message_response(content="first"),
            message_response(tool_calls=[second_tool]),
            message_response(content="second"),
        ]
    )
    state = ConversationState(user_query="question")
    tool_calls = []

    def execute(name, arguments):
        tool_calls.append((name, arguments))
        return add_reference_result(state, name)

    first = list(
        run_agentic_loop(
            llm,
            state,
            execute,
            max_calls=2,
            token_threshold=10_000,
            token_warning_ratio=0.8,
            status_writer=lambda status: None,
        )
    )
    second = list(
        run_agentic_loop(
            llm,
            state,
            execute,
            max_calls=2,
            token_threshold=10_000,
            token_warning_ratio=0.8,
            status_writer=lambda status: None,
        )
    )

    system_prompts = [
        message
        for message in state.messages
        if message.get("role") == "system" and message.get("content") == SYSTEM_PROMPT
    ]
    assert first == ["first"]
    assert second == ["second"]
    assert tool_calls == [
        ("search", {"queries": ["first"]}),
        ("search", {"queries": ["second"]}),
    ]
    assert len(system_prompts) == 1
    assert state.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_run_agentic_loop_handles_dict_shaped_tool_calls():
    llm = FakeLLM(
        responses=[
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "dict-call",
                                    "function": {
                                        "name": "find",
                                        "arguments": '{"reference_id": "ref", "patterns": ["a"]}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "done", "tool_calls": None}}]},
        ]
    )
    state = ConversationState(user_query="question")
    tool_calls = []

    chunks = list(
        run_agentic_loop(
            llm,
            state,
            lambda name, arguments: tool_calls.append((name, arguments))
            or add_reference_result(state, name, reference_id="ref", content="found"),
            max_calls=2,
            token_threshold=10_000,
            token_warning_ratio=0.8,
            status_writer=lambda status: None,
        )
    )

    assert chunks == ["done"]
    assert tool_calls == [("find", {"reference_id": "ref", "patterns": ["a"]})]
    assert state.messages[-2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "dict-call",
                "type": "function",
                "function": {
                    "name": "find",
                    "arguments": '{"reference_id": "ref", "patterns": ["a"]}',
                },
            }
        ],
    }
    assert state.messages[-1] == {
        "role": "tool",
        "content": "found",
        "tool_call_id": "dict-call",
        "name": "find",
    }


def test_run_agentic_loop_passes_error_dict_for_invalid_tool_arguments():
    invalid_arguments = [
        ("invalid-json", "{"),
        ("none", None),
        ("empty", ""),
        ("array", '["not", "object"]'),
        ("scalar", "42"),
    ]
    responses = [
        message_response(
            tool_calls=[
                tool_call(call_id, "search", arguments)
                for call_id, arguments in invalid_arguments
            ]
        ),
        message_response(content="done"),
    ]
    llm = FakeLLM(responses=responses)
    state = ConversationState(user_query="question")
    tool_calls = []

    chunks = list(
        run_agentic_loop(
            llm,
            state,
            lambda name, arguments: tool_calls.append((name, arguments)) or "tool result",
            max_calls=2,
            token_threshold=10_000,
            token_warning_ratio=0.8,
            status_writer=lambda status: None,
        )
    )

    assert chunks == ["done"]
    assert tool_calls == [
        ("search", {"_error": "invalid tool arguments"}),
        ("search", {"_error": "invalid tool arguments"}),
        ("search", {"_error": "invalid tool arguments"}),
        ("search", {"_error": "tool arguments must be a JSON object"}),
        ("search", {"_error": "tool arguments must be a JSON object"}),
    ]
    tool_messages = [message for message in state.messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "invalid-json",
        "none",
        "empty",
        "array",
        "scalar",
    ]


def test_run_agentic_loop_adds_provider_tool_call_id_to_tool_executor_messages():
    tool = tool_call("call-1", "search", '{"queries": ["alpha"]}')
    llm = FakeLLM(
        responses=[
            message_response(tool_calls=[tool]),
            message_response(content="done"),
        ]
    )
    state = ConversationState(user_query="question")

    def execute(name, arguments):
        state.add_tool_result(
            name,
            "stored tool result",
            metadata={"reference_ids": ["turn0search0"]},
        )
        return "stored tool result"

    chunks = list(
        run_agentic_loop(
            llm,
            state,
            execute,
            max_calls=2,
            token_threshold=10_000,
            token_warning_ratio=0.8,
            status_writer=lambda status: None,
        )
    )

    tool_messages = [message for message in state.messages if message["role"] == "tool"]
    assert chunks == ["done"]
    assert tool_messages == [
        {
            "role": "tool",
            "name": "search",
            "content": "stored tool result",
            "tool_call_id": "call-1",
        }
    ]


def test_run_agentic_loop_compresses_state_when_token_threshold_is_reached():
    tool = tool_call("call-1", "search", '{"queries": ["current"]}')
    llm = FakeLLM(
        responses=[
            message_response(tool_calls=[tool]),
            message_response(content="done"),
        ]
    )
    state = ConversationState(user_query="question")
    state.references["keep"] = SimpleNamespace()
    state.add_tool_result(
        "search",
        "retained full result content",
        metadata={"reference_ids": ["keep"]},
    )
    state.add_tool_result(
        "search",
        "unrelated full result content",
        metadata={"reference_ids": ["drop"]},
    )
    tool_calls = []

    chunks = list(
        run_agentic_loop(
            llm,
            state,
            lambda name, arguments: tool_calls.append((name, arguments))
            or (
                "summary"
                if name == "summarize"
                else add_reference_result(
                    state,
                    name,
                    reference_id="current",
                    content="current search result",
                )
            ),
            max_calls=2,
            token_threshold=0,
            token_warning_ratio=0.8,
            status_writer=lambda status: None,
        )
    )

    tool_messages = [message for message in state.messages if message["role"] == "tool"]
    assert chunks == ["done"]
    assert state.tool_results[0].content == "retained full result content"
    assert state.tool_results[1].content == (
        "[compressed search result unrelated to retained references]"
    )
    assert [message["content"] for message in tool_messages[:2]] == [
        "retained full result content",
        "[compressed search result unrelated to retained references]",
    ]
    assert tool_messages[-1]["content"] == "current search result"
    assert tool_calls == [
        ("summarize", {"candidate_reference_ids": ["keep"]}),
        ("search", {"queries": ["current"]}),
        ("summarize", {"candidate_reference_ids": ["keep", "current"]}),
    ]
