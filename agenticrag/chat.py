from __future__ import annotations

import json
from typing import Any, Protocol

from agenticrag.prompts import QUERY_REWRITE_PROMPT
from agenticrag.state import ConversationState

Message = dict[str, str]


class SupportsComplete(Protocol):
    def complete(self, *, messages: list[Message]) -> str: ...


def parse_rewrite_response(text: str, fallback: str) -> str:
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(payload, dict):
        return fallback

    query = payload.get("query")
    if not isinstance(query, str):
        return fallback

    query = query.strip()
    return query or fallback


def _reference_summary(state: ConversationState, limit: int = 12) -> str:
    references = list(state.references.values())[-limit:]
    if not references:
        return "No Reference IDs yet."

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
        return "No prior conversation."

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
    try:
        response = llm_client.complete(messages=build_rewrite_messages(state, user_input))
    except Exception:
        return user_input

    return parse_rewrite_response(response, user_input)
