from __future__ import annotations

import json
import re
from typing import Any, Protocol

from agenticrag.prompts import QUERY_REWRITE_PROMPT
from agenticrag.state import ConversationState

Message = dict[str, str]


class SupportsComplete(Protocol):
    def complete(self, *, messages: list[Message]) -> str: ...


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
