from __future__ import annotations

import json
import re
from typing import Any, Iterator, Protocol

from agenticrag.loop import execute_retrieval_tool, run_agentic_loop
from agenticrag.prompts import CHAT_SIMPLE_RAG_PROMPT, QUERY_REWRITE_PROMPT
from agenticrag.state import ConversationState
from agenticrag.switcher import classify_query

Message = dict[str, str]

CHAT_HELP = """Commands:
/help  Show this help.
/reset Reset the conversation session.
/exit  Exit chat.
/quit  Exit chat."""


class SupportsComplete(Protocol):
    def complete(self, *, messages: list[Message]) -> str: ...


class SupportsStream(Protocol):
    def stream(self, messages: list[Message]) -> Iterator[str]: ...


def _json_candidates(text: str) -> list[str]:
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.append(text)
    candidates.extend(re.findall(r"\{.*?\}", text, flags=re.DOTALL))
    return candidates


def parse_rewrite_response(text: str, fallback: str) -> str:
    for candidate in _json_candidates(text):
        try:
            payload: Any = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue

        query = payload.get("query")
        if not isinstance(query, str):
            continue

        query = query.strip()
        if query:
            return query

    return fallback


def _reference_summary(state: ConversationState, limit: int = 12) -> str:
    references = list(state.references.values())[-limit:]
    if not references:
        return ""

    lines = []
    for reference in references:
        chunk = reference.chunk
        lines.append(f"- {reference.reference_id}: {chunk.title} ({chunk.path.as_posix()})")
    return "\n".join(lines)


def _history_summary(state: ConversationState, limit: int) -> str:
    messages = [
        message
        for message in state.messages
        if message.get("role") != "tool" and message.get("content") is not None
    ][-limit:]
    if not messages:
        return ""

    lines = []
    for message in messages:
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", ""))[:800]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_rewrite_messages(
    state: ConversationState,
    user_input: str,
    history_limit: int = 8,
) -> list[Message]:
    content = "\n\n".join(
        [
            "Recent conversation:",
            _history_summary(state, history_limit),
            "Available Reference IDs:",
            _reference_summary(state),
            "Current user question:",
            user_input,
        ]
    )
    return [
        {"role": "system", "content": QUERY_REWRITE_PROMPT},
        {"role": "user", "content": content},
    ]


def rewrite_query(
    llm_client: SupportsComplete,
    state: ConversationState,
    user_input: str,
) -> str:
    messages = build_rewrite_messages(state, user_input)
    try:
        response = llm_client.complete(messages=messages)
        return parse_rewrite_response(response, user_input)
    except Exception:
        return user_input


def stream_simple_chat(
    llm_client: SupportsStream,
    raw_query: str,
    rewritten_query: str,
    search_context: str,
) -> Iterator[str]:
    messages = [
        {"role": "system", "content": CHAT_SIMPLE_RAG_PROMPT},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "Raw user question:",
                    raw_query,
                    "Rewritten self-contained question:",
                    rewritten_query,
                    "Search results/context:",
                    search_context,
                ]
            ),
        },
    ]
    yield from llm_client.stream(messages)


class ChatSession:
    def __init__(
        self,
        *,
        llm_client: SupportsComplete,
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
        self.state = self._new_state()
        self._attach_tools_state()

    def reset(self) -> None:
        self.state = self._new_state()
        self._attach_tools_state()

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
            search_context = self.tools.search([rewritten_query])
            chunks: list[str] = []
            for chunk in stream_simple_chat(
                self.llm_client,
                user_input,
                rewritten_query,
                search_context,
            ):
                chunks.append(chunk)
                yield chunk
            self.state.add_message("assistant", "".join(chunks))
            return

        self.state.messages[-1]["content"] = "\n".join(
            [
                f"Raw user question: {user_input}",
                f"Rewritten self-contained question: {rewritten_query}",
            ]
        )
        chat_visible_messages = [dict(message) for message in self.state.messages]
        chunks = []
        completed = False
        try:
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
            completed = True
        finally:
            self.state.messages = [dict(message) for message in chat_visible_messages]
        if not completed:
            return
        self.state.add_message("assistant", "".join(chunks))

    @staticmethod
    def _new_state() -> ConversationState:
        state = ConversationState(user_query="")
        if state.messages == [{"role": "user", "content": ""}]:
            state.messages.clear()
        return state

    def _attach_tools_state(self) -> None:
        if hasattr(self.tools, "state"):
            self.tools.state = self.state


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
    state = ConversationState(user_query="")
    tools = RetrievalTools(
        retriever=retriever,
        state=state,
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
