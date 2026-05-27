# AgenticRAG Multi-Turn Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `python main.py chat`, an interactive multi-turn AgenticRAG session with query rewriting, persistent in-memory `ConversationState`, stable Reference IDs, and streaming answers.

**Architecture:** Add a focused `agenticrag/chat.py` module for chat-session orchestration and REPL commands. Keep `agenticrag/loop.py` as the one-turn agentic execution engine, but extract a shared retrieval-tool dispatcher so `ask` and `chat` use the same validation behavior. Update `main.py` only for parser dispatch.

**Tech Stack:** Python 3.11, pytest, argparse, DeepSeek OpenAI-compatible chat client, SiliconFlow embeddings, Chroma retriever, existing `ConversationState` and `RetrievalTools`.

---

## File Structure

- `agenticrag/chat.py`: new multi-turn chat module. Owns query rewrite helpers, `ChatSession`, and `run_chat`.
- `agenticrag/prompts.py`: add `QUERY_REWRITE_PROMPT` and `CHAT_SIMPLE_RAG_PROMPT`.
- `agenticrag/loop.py`: extract shared `execute_retrieval_tool` and update `run_ask` to use it.
- `main.py`: add `chat` subcommand and dispatch to `agenticrag.chat.run_chat`.
- `tests/test_chat.py`: new focused tests for rewrite, `ChatSession`, REPL commands, streaming, reset, and reference retention.
- `tests/test_cli.py`: extend parser and dispatch tests for `chat`.
- `README.md`: document `python main.py chat` after implementation is verified.

---

### Task 1: Query Rewrite Prompts And Parsing

**Files:**
- Modify: `agenticrag/prompts.py`
- Create: `agenticrag/chat.py`
- Test: `tests/test_chat.py`

- [ ] **Step 1: Write failing rewrite tests**

Create `tests/test_chat.py`:

```python
from types import SimpleNamespace

from agenticrag.chat import build_rewrite_messages
from agenticrag.chat import parse_rewrite_response
from agenticrag.chat import rewrite_query
from agenticrag.prompts import QUERY_REWRITE_PROMPT
from agenticrag.state import ConversationState


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return self.response


def test_parse_rewrite_response_extracts_query():
    assert parse_rewrite_response('{"query": "请详细说明第二个模块"}', "原问题") == (
        "请详细说明第二个模块"
    )


def test_parse_rewrite_response_falls_back_for_bad_json():
    assert parse_rewrite_response("not json", "第二个呢") == "第二个呢"
    assert parse_rewrite_response('{"query": ""}', "第二个呢") == "第二个呢"
    assert parse_rewrite_response('{"query": 123}', "第二个呢") == "第二个呢"


def test_build_rewrite_messages_include_recent_history_and_refs():
    state = ConversationState(user_query="")
    state.add_message("user", "AgenticRAG 的核心模块有哪些？")
    state.add_message("assistant", "包括 Query Switcher 和 Agentic Loop。")
    state.references["turn0search0"] = SimpleNamespace(
        chunk=SimpleNamespace(title="Design", path=SimpleNamespace(as_posix=lambda: "docs/design.md"))
    )

    messages = build_rewrite_messages(state, "第二个模块详细说说", history_limit=4)

    assert messages[0] == {"role": "system", "content": QUERY_REWRITE_PROMPT}
    assert "第二个模块详细说说" in messages[1]["content"]
    assert "AgenticRAG 的核心模块有哪些？" in messages[1]["content"]
    assert "turn0search0" in messages[1]["content"]
    assert "docs/design.md" in messages[1]["content"]


def test_rewrite_query_calls_llm_and_falls_back_on_empty_result():
    state = ConversationState(user_query="")
    llm = FakeLLM('{"query": "AgenticRAG 的 Agentic Loop 详细说明"}')

    assert rewrite_query(llm, state, "第二个呢") == "AgenticRAG 的 Agentic Loop 详细说明"
    assert llm.messages[0]["content"] == QUERY_REWRITE_PROMPT

    assert rewrite_query(FakeLLM("{}"), state, "第二个呢") == "第二个呢"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chat.py -v
```

Expected: FAIL with missing `agenticrag.chat`.

- [ ] **Step 3: Add rewrite prompts**

Append to `agenticrag/prompts.py`:

```python
QUERY_REWRITE_PROMPT = """Rewrite the user's current question into a self-contained question for a multi-turn AgenticRAG session.
Use the recent conversation and available Reference IDs only to resolve context, pronouns, ellipsis, and follow-up references.
Return only JSON in this shape: {"query": "..."}.
If the current question is already self-contained, return it unchanged in the query field.
Do not answer the question.
"""

CHAT_SIMPLE_RAG_PROMPT = """请基于给定检索片段回答用户问题。
你会看到原始用户问题、改写后的自包含问题和检索结果。
回答应面向原始用户问题，必要时利用改写后的问题消解上下文。
必须引用片段中的 Reference ID。如果证据不足，请明确说明。
"""
```

- [ ] **Step 4: Implement rewrite helpers**

Create `agenticrag/chat.py`:

```python
from __future__ import annotations

import json
from typing import Any

from agenticrag.prompts import QUERY_REWRITE_PROMPT
from agenticrag.state import ConversationState


def parse_rewrite_response(text: str, fallback: str) -> str:
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(payload, dict):
        return fallback
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return fallback
    return query.strip()


def _reference_summary(state: ConversationState, limit: int = 12) -> str:
    lines: list[str] = []
    for reference_id, reference in list(state.references.items())[-limit:]:
        chunk = reference.chunk
        lines.append(
            f"- {reference_id}: {chunk.title} ({chunk.path.as_posix()})"
        )
    return "\n".join(lines) if lines else "No Reference IDs yet."


def _history_summary(state: ConversationState, limit: int) -> str:
    recent = state.messages[-limit:]
    lines: list[str] = []
    for message in recent:
        role = message.get("role", "")
        if role == "tool":
            continue
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:800]}")
    return "\n".join(lines) if lines else "No prior conversation."


def build_rewrite_messages(
    state: ConversationState,
    user_input: str,
    history_limit: int = 8,
) -> list[dict[str, str]]:
    content = (
        "Recent conversation:\n"
        f"{_history_summary(state, history_limit)}\n\n"
        "Available Reference IDs:\n"
        f"{_reference_summary(state)}\n\n"
        "Current user question:\n"
        f"{user_input}"
    )
    return [
        {"role": "system", "content": QUERY_REWRITE_PROMPT},
        {"role": "user", "content": content},
    ]


def rewrite_query(llm_client: Any, state: ConversationState, user_input: str) -> str:
    try:
        response = llm_client.complete(
            build_rewrite_messages(state, user_input)
        )
    except Exception:
        return user_input
    return parse_rewrite_response(response, user_input)
```

- [ ] **Step 5: Run rewrite tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chat.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add agenticrag/prompts.py agenticrag/chat.py tests/test_chat.py
git commit -m "feat: add chat query rewriting"
```

---

### Task 2: Shared Retrieval Tool Dispatcher

**Files:**
- Modify: `agenticrag/loop.py`
- Test: `tests/test_loop.py`

- [ ] **Step 1: Write failing dispatcher tests**

Append to `tests/test_loop.py`:

```python
from agenticrag.loop import execute_retrieval_tool


class FakeRetrievalTools:
    def __init__(self):
        self.calls = []

    def search(self, queries):
        self.calls.append(("search", queries))
        return "search result"

    def find(self, reference_id, patterns):
        self.calls.append(("find", reference_id, patterns))
        return "find result"

    def open(self, reference_id, line_number=0):
        self.calls.append(("open", reference_id, line_number))
        return "open result"

    def summarize(self, candidate_reference_ids):
        self.calls.append(("summarize", candidate_reference_ids))
        return "summary result"


def test_execute_retrieval_tool_dispatches_all_tools():
    tools = FakeRetrievalTools()

    assert execute_retrieval_tool(tools, "search", {"queries": ["q"]}) == "search result"
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
        "summarize",
        {"candidate_reference_ids": ["turn0search0"]},
    ) == "summary result"

    assert tools.calls == [
        ("search", ["q"]),
        ("find", "turn0search0", ["agent"]),
        ("open", "turn0search0", 7),
        ("summarize", ["turn0search0"]),
    ]


def test_execute_retrieval_tool_returns_tool_errors_for_bad_arguments():
    tools = FakeRetrievalTools()

    assert execute_retrieval_tool(tools, "search", {"_error": "invalid tool arguments"}) == (
        "[tool error] search: invalid tool arguments"
    )
    assert execute_retrieval_tool(tools, "search", {"queries": "q"}).startswith(
        "[tool error] search:"
    )
    assert execute_retrieval_tool(tools, "find", {"patterns": []}).startswith(
        "[tool error] find:"
    )
    assert execute_retrieval_tool(
        tools,
        "open",
        {"reference_id": "ref", "line_number": "7"},
    ).startswith("[tool error] open:")
    assert execute_retrieval_tool(tools, "missing", {}) == (
        "[tool error] missing: unknown tool"
    )
```

- [ ] **Step 2: Run dispatcher tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_loop.py::test_execute_retrieval_tool_dispatches_all_tools tests/test_loop.py::test_execute_retrieval_tool_returns_tool_errors_for_bad_arguments -v
```

Expected: FAIL with missing `execute_retrieval_tool`.

- [ ] **Step 3: Extract dispatcher in `agenticrag/loop.py`**

Add above `run_agentic_loop`:

```python
def execute_retrieval_tool(tools: Any, name: str, arguments: dict[str, Any]) -> str:
    if "_error" in arguments:
        return f"[tool error] {name}: {arguments['_error']}"

    try:
        if name == "search":
            queries = arguments.get("queries")
            if not isinstance(queries, list):
                return "[tool error] search: queries must be a list"
            return tools.search(queries)
        if name == "find":
            reference_id = arguments.get("reference_id")
            patterns = arguments.get("patterns")
            if not isinstance(reference_id, str):
                return "[tool error] find: reference_id must be a string"
            if not isinstance(patterns, list):
                return "[tool error] find: patterns must be a list"
            return tools.find(reference_id, patterns)
        if name == "open":
            reference_id = arguments.get("reference_id")
            line_number = arguments.get("line_number", 0)
            if not isinstance(reference_id, str):
                return "[tool error] open: reference_id must be a string"
            if not isinstance(line_number, int):
                return "[tool error] open: line_number must be an integer"
            return tools.open(reference_id, line_number=line_number)
        if name == "summarize":
            candidate_reference_ids = arguments.get("candidate_reference_ids")
            if not isinstance(candidate_reference_ids, list):
                return "[tool error] summarize: candidate_reference_ids must be a list"
            return tools.summarize(candidate_reference_ids)
    except Exception as exc:
        return f"[tool error] {name}: {exc}"

    return f"[tool error] {name}: unknown tool"
```

Update `run_ask` by replacing its nested `execute_tool` body with:

```python
    def execute_tool(name: str, arguments: dict[str, Any]) -> str:
        return execute_retrieval_tool(tools, name, arguments)
```

- [ ] **Step 4: Run dispatcher and CLI tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_loop.py tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add agenticrag/loop.py tests/test_loop.py tests/test_cli.py
git commit -m "refactor: share retrieval tool dispatch"
```

---

### Task 3: ChatSession Simple Route

**Files:**
- Modify: `agenticrag/chat.py`
- Test: `tests/test_chat.py`

- [ ] **Step 1: Write failing ChatSession simple route tests**

Append to `tests/test_chat.py`:

```python
from agenticrag.chat import ChatSession


class FakeTools:
    def __init__(self):
        self.search_calls = []
        self.state = None

    def search(self, queries):
        self.search_calls.append(queries)
        self.state.references["turn0search0"] = SimpleNamespace(
            chunk=SimpleNamespace(title="Doc", path=SimpleNamespace(as_posix=lambda: "docs/doc.md"))
        )
        return "[turn0search0] context"


def test_chat_session_simple_route_streams_and_records_answer(monkeypatch):
    state_seen_by_rewriter = []
    tools = FakeTools()
    llm = FakeLLM('{"query": "AgenticRAG 核心模块"}')

    def fake_rewrite_query(llm_client, state, user_input):
        state_seen_by_rewriter.append((list(state.messages), user_input))
        return "AgenticRAG 核心模块"

    def fake_classify_query(llm_client, query):
        assert query == "AgenticRAG 核心模块"
        return "simple"

    def fake_stream_simple_chat(llm_client, raw_query, rewritten_query, context):
        assert raw_query == "它有哪些核心模块？"
        assert rewritten_query == "AgenticRAG 核心模块"
        assert context == "[turn0search0] context"
        yield "流式"
        yield "回答"

    monkeypatch.setattr("agenticrag.chat.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("agenticrag.chat.classify_query", fake_classify_query)
    monkeypatch.setattr("agenticrag.chat.stream_simple_chat", fake_stream_simple_chat)

    session = ChatSession(
        llm_client=llm,
        tools=tools,
        max_calls=3,
        token_threshold=10_000,
        token_warning_ratio=0.8,
    )
    tools.state = session.state

    chunks = list(session.answer_turn("它有哪些核心模块？"))

    assert chunks == ["流式", "回答"]
    assert tools.search_calls == [["AgenticRAG 核心模块"]]
    assert session.state.messages[-2] == {"role": "user", "content": "它有哪些核心模块？"}
    assert session.state.messages[-1] == {"role": "assistant", "content": "流式回答"}
    assert "turn0search0" in session.state.references
    assert state_seen_by_rewriter[0][0][-1] == {
        "role": "user",
        "content": "它有哪些核心模块？",
    }
```

- [ ] **Step 2: Run simple route test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chat.py::test_chat_session_simple_route_streams_and_records_answer -v
```

Expected: FAIL with missing `ChatSession`.

- [ ] **Step 3: Add chat simple prompt and stream helper**

In `agenticrag/chat.py`, import `CHAT_SIMPLE_RAG_PROMPT` and add:

```python
from collections.abc import Iterator

from agenticrag.prompts import CHAT_SIMPLE_RAG_PROMPT


def stream_simple_chat(
    llm_client: Any,
    raw_query: str,
    rewritten_query: str,
    search_context: str,
) -> Iterator[str]:
    messages = [
        {"role": "system", "content": CHAT_SIMPLE_RAG_PROMPT},
        {
            "role": "user",
            "content": (
                f"原始问题：\n{raw_query}\n\n"
                f"改写后的自包含问题：\n{rewritten_query}\n\n"
                f"检索结果：\n{search_context}"
            ),
        },
    ]
    yield from llm_client.stream(messages)
```

- [ ] **Step 4: Implement `ChatSession` simple path**

In `agenticrag/chat.py`, add:

```python
from agenticrag.state import ConversationState
from agenticrag.switcher import classify_query


class ChatSession:
    def __init__(
        self,
        *,
        llm_client: Any,
        tools: Any,
        max_calls: int,
        token_threshold: int,
        token_warning_ratio: float,
    ) -> None:
        self.llm_client = llm_client
        self.tools = tools
        self.max_calls = max_calls
        self.token_threshold = token_threshold
        self.token_warning_ratio = token_warning_ratio
        self.state = ConversationState(user_query="")

    def reset(self) -> None:
        self.state = ConversationState(user_query="")
        if hasattr(self.tools, "state"):
            self.tools.state = self.state

    def answer_turn(self, user_input: str) -> Iterator[str]:
        self.state.add_message("user", user_input)
        rewritten_query = rewrite_query(self.llm_client, self.state, user_input)
        try:
            route = classify_query(self.llm_client, rewritten_query)
        except Exception:
            route = "complex"

        if route == "simple":
            context = self.tools.search([rewritten_query])
            chunks: list[str] = []
            for chunk in stream_simple_chat(
                self.llm_client,
                user_input,
                rewritten_query,
                context,
            ):
                chunks.append(chunk)
                yield chunk
            self.state.add_message("assistant", "".join(chunks))
            return

        raise NotImplementedError("complex chat route is implemented in Task 4")
```

- [ ] **Step 5: Run chat tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chat.py -v
```

Expected: PASS for rewrite and simple route tests.

- [ ] **Step 6: Commit**

```powershell
git add agenticrag/chat.py tests/test_chat.py
git commit -m "feat: add chat session simple turns"
```

---

### Task 4: ChatSession Complex Route And Reference Retention

**Files:**
- Modify: `agenticrag/chat.py`
- Test: `tests/test_chat.py`

- [ ] **Step 1: Write failing complex route tests**

Append to `tests/test_chat.py`:

```python
def test_chat_session_complex_route_streams_status_and_records_answer(monkeypatch):
    tools = FakeTools()
    llm = FakeLLM('{"query": "对比四个工具"}')
    statuses = []

    monkeypatch.setattr("agenticrag.chat.rewrite_query", lambda llm_client, state, user_input: "对比四个工具")
    monkeypatch.setattr("agenticrag.chat.classify_query", lambda llm_client, query: "complex")

    def fake_run_agentic_loop(
        llm_client,
        state,
        tool_executor,
        max_calls,
        token_threshold,
        token_warning_ratio,
        status_writer,
    ):
        assert state.messages[-1]["content"] == (
            "原始问题：请对比它们\n改写后的自包含问题：对比四个工具"
        )
        status_writer("[tool] search")
        statuses.append("status-called")
        assert tool_executor("search", {"queries": ["对比四个工具"]}) == "[turn0search0] context"
        yield "复杂"
        yield "回答"

    monkeypatch.setattr("agenticrag.chat.run_agentic_loop", fake_run_agentic_loop)

    session = ChatSession(
        llm_client=llm,
        tools=tools,
        max_calls=4,
        token_threshold=9000,
        token_warning_ratio=0.8,
    )
    tools.state = session.state

    chunks = list(session.answer_turn("请对比它们", status_writer=statuses.append))

    assert chunks == ["复杂", "回答"]
    assert "[tool] search" in statuses
    assert session.state.messages[-1] == {"role": "assistant", "content": "复杂回答"}
    assert "turn0search0" in session.state.references


def test_chat_session_reset_clears_state_and_refs():
    tools = FakeTools()
    session = ChatSession(
        llm_client=FakeLLM("{}"),
        tools=tools,
        max_calls=1,
        token_threshold=100,
        token_warning_ratio=0.8,
    )
    tools.state = session.state
    session.state.add_message("user", "hello")
    session.state.references["turn0search0"] = SimpleNamespace(chunk=SimpleNamespace())

    session.reset()

    assert session.state.messages == [{"role": "user", "content": ""}]
    assert session.state.references == {}
    assert tools.state is session.state
```

- [ ] **Step 2: Run complex route tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chat.py::test_chat_session_complex_route_streams_status_and_records_answer tests/test_chat.py::test_chat_session_reset_clears_state_and_refs -v
```

Expected: FAIL because complex route raises `NotImplementedError` or reset behavior is incomplete.

- [ ] **Step 3: Implement complex route**

Update `agenticrag/chat.py` imports:

```python
from agenticrag.loop import execute_retrieval_tool
from agenticrag.loop import run_agentic_loop
```

Update `ChatSession.answer_turn` signature and complex branch:

```python
    def answer_turn(
        self,
        user_input: str,
        status_writer: Any | None = None,
    ) -> Iterator[str]:
        writer = status_writer or (lambda text: None)
        self.state.add_message("user", user_input)
        rewritten_query = rewrite_query(self.llm_client, self.state, user_input)
        try:
            route = classify_query(self.llm_client, rewritten_query)
        except Exception:
            route = "complex"

        if route == "simple":
            context = self.tools.search([rewritten_query])
            chunks: list[str] = []
            for chunk in stream_simple_chat(
                self.llm_client,
                user_input,
                rewritten_query,
                context,
            ):
                chunks.append(chunk)
                yield chunk
            self.state.add_message("assistant", "".join(chunks))
            return

        self.state.add_message(
            "user",
            f"原始问题：{user_input}\n改写后的自包含问题：{rewritten_query}",
        )
        chunks = []
        for chunk in run_agentic_loop(
            self.llm_client,
            self.state,
            lambda name, arguments: execute_retrieval_tool(
                self.tools,
                name,
                arguments,
            ),
            max_calls=self.max_calls,
            token_threshold=self.token_threshold,
            token_warning_ratio=self.token_warning_ratio,
            status_writer=writer,
        ):
            chunks.append(chunk)
            yield chunk
        self.state.add_message("assistant", "".join(chunks))
```

- [ ] **Step 4: Run chat tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chat.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add agenticrag/chat.py tests/test_chat.py
git commit -m "feat: support complex chat turns"
```

---

### Task 5: Chat REPL And CLI Dispatch

**Files:**
- Modify: `agenticrag/chat.py`
- Modify: `main.py`
- Modify: `tests/test_cli.py`
- Test: `tests/test_chat.py`

- [ ] **Step 1: Write failing REPL and CLI tests**

Append to `tests/test_chat.py`:

```python
def test_run_chat_handles_help_reset_and_exit(monkeypatch, capsys):
    from agenticrag import chat

    class FakeSession:
        def __init__(self):
            self.reset_calls = 0

        def answer_turn(self, user_input, status_writer=None):
            yield f"answer:{user_input}"

        def reset(self):
            self.reset_calls += 1

    session = FakeSession()
    inputs = iter(["/help", "", "hello", "/reset", "/exit"])

    monkeypatch.setattr(chat, "create_chat_session", lambda: session)
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    assert chat.run_chat() == 0

    output = capsys.readouterr().out
    assert "AgenticRAG chat" in output
    assert "/reset" in output
    assert "answer:hello" in output
    assert "Session reset." in output
    assert session.reset_calls == 1
```

Append to `tests/test_cli.py`:

```python
def test_build_parser_parses_chat_command():
    parser = cli.build_parser()

    chat_args = parser.parse_args(["chat"])

    assert chat_args.command == "chat"


def test_main_dispatches_chat(monkeypatch):
    calls = []

    def fake_run_chat():
        calls.append("chat")
        return 0

    monkeypatch.setattr("agenticrag.chat.run_chat", fake_run_chat, raising=False)

    assert cli.main(["chat"]) == 0
    assert calls == ["chat"]
```

- [ ] **Step 2: Run new tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chat.py::test_run_chat_handles_help_reset_and_exit tests/test_cli.py::test_build_parser_parses_chat_command tests/test_cli.py::test_main_dispatches_chat -v
```

Expected: FAIL because `run_chat`, `create_chat_session`, and CLI `chat` are missing.

- [ ] **Step 3: Implement chat runtime factory and REPL**

Append to `agenticrag/chat.py`:

```python
def create_chat_session() -> ChatSession:
    from agenticrag.config import load_config
    from agenticrag.embeddings import SiliconFlowEmbeddingClient
    from agenticrag.llm import DeepSeekClient
    from agenticrag.retriever import ChromaRetriever
    from agenticrag.tools.retrieval import RetrievalTools

    config = load_config()
    llm_client = DeepSeekClient(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        model=config.deepseek_model,
    )
    embedding_client = SiliconFlowEmbeddingClient(
        api_key=config.siliconflow_api_key,
        base_url=config.siliconflow_base_url,
        model=config.siliconflow_embedding_model,
    )
    retriever = ChromaRetriever(config.chroma_dir, embedding_client)
    placeholder_state = ConversationState(user_query="")
    tools = RetrievalTools(
        retriever=retriever,
        state=placeholder_state,
        source_cache_dir=config.source_cache_dir,
    )
    session = ChatSession(
        llm_client=llm_client,
        tools=tools,
        max_calls=config.max_calls,
        token_threshold=config.token_threshold,
        token_warning_ratio=config.token_warning_ratio,
    )
    tools.state = session.state
    return session


CHAT_HELP = """Commands:
  /help   Show this help
  /reset  Clear the current chat session
  /exit   Exit chat
  /quit   Exit chat
"""


def run_chat() -> int:
    session = create_chat_session()
    print("AgenticRAG chat. Type /help for commands, /exit to quit.")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(CHAT_HELP)
            continue
        if user_input == "/reset":
            session.reset()
            print("Session reset.")
            continue

        try:
            for chunk in session.answer_turn(
                user_input,
                status_writer=lambda text: print(text, flush=True),
            ):
                print(chunk, end="", flush=True)
            print()
        except Exception as exc:
            print(f"[error] {exc}")
    return 0
```

- [ ] **Step 4: Add CLI `chat` parser and dispatch**

Update `main.py` `build_parser`:

```python
    subparsers.add_parser("chat")
```

Update `main` dispatch:

```python
        if args.command == "chat":
            from agenticrag import chat as chat_module

            return chat_module.run_chat()
```

- [ ] **Step 5: Run chat and CLI tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chat.py tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add agenticrag/chat.py main.py tests/test_chat.py tests/test_cli.py
git commit -m "feat: add interactive chat command"
```

---

### Task 6: README And Focused Verification

**Files:**
- Modify: `README.md`
- Modify only implementation files needed to fix verification failures.

- [ ] **Step 1: Update README with chat usage**

Add after the Ask section in `README.md`:

```markdown
## Chat

```powershell
python main.py chat
```

Use `chat` for multi-turn follow-up questions in one terminal session.
The session keeps conversation context and Reference IDs until you exit or run `/reset`.

Available chat commands:

- `/help`: show commands
- `/reset`: clear the current in-memory session
- `/exit` or `/quit`: exit chat

Answers stream to the terminal. Complex AgenticRAG turns may print compact `[tool] ...` status lines before the final answer.
```

- [ ] **Step 2: Run full focused test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run CLI help checks**

Run:

```powershell
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe main.py chat --help
```

Expected: both commands print argparse help text and exit with code 0.

- [ ] **Step 4: Run optional manual chat smoke if real keys are available**

If `.env` contains real `DEEPSEEK_API_KEY` and `SILICONFLOW_API_KEY`, run:

```powershell
.\.venv\Scripts\python.exe main.py chat
```

Manual inputs:

```text
AgenticRAG 的核心模块有哪些？
第二个模块详细说说
/exit
```

Expected: both answers stream; the second answer resolves the follow-up using history.

Do not print API keys. If keys are missing, record that manual smoke was skipped.

- [ ] **Step 5: Commit docs and verification fixes**

```powershell
git add README.md agenticrag main.py tests
git commit -m "docs: add multi-turn chat usage"
```

If README was already committed in a previous task and no files changed here, do not create an empty commit.

---

## Self-Review

Spec coverage:

- `python main.py chat`: Task 5.
- Basic commands `/help`, `/reset`, `/exit`, `/quit`, empty input ignore: Task 5.
- Long-lived `ConversationState`: Tasks 3 and 4.
- Query rewriting before switcher: Tasks 1, 3, and 4.
- Streaming answers in simple and complex routes: Tasks 3, 4, and 5.
- Stable Reference IDs across turns: Task 4 tests and persistent session state.
- Reusing existing agentic loop and retrieval tools: Tasks 2 and 4.
- README and verification: Task 6.

Completeness scan:

- This plan contains concrete implementation steps, exact files, verification commands, and commit points.

Type consistency:

- `ChatSession.answer_turn(user_input: str, status_writer: Any | None = None) -> Iterator[str]` is introduced before REPL use.
- `execute_retrieval_tool(tools, name, arguments)` is introduced before `ChatSession` complex route uses it.
- `create_chat_session() -> ChatSession` is introduced before `run_chat()` uses it.
