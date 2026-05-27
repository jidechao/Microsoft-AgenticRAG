from __future__ import annotations

import json
from typing import Any, Callable, Iterator

from agenticrag.prompts import FORCE_FINAL_ANSWER_PROMPT
from agenticrag.prompts import SIMPLE_RAG_PROMPT
from agenticrag.prompts import SYSTEM_PROMPT
from agenticrag.state import ConversationState
from agenticrag.tools.schemas import TOOL_SCHEMAS

QUESTION_LABEL = "\u95ee\u9898\uff1a"
SEARCH_RESULTS_LABEL = "\u68c0\u7d22\u7ed3\u679c\uff1a"
INVALID_TOOL_ARGUMENTS = {"_error": "invalid tool arguments"}
NON_OBJECT_TOOL_ARGUMENTS = {"_error": "tool arguments must be a JSON object"}


def should_force_completion(call_index: int, max_calls: int) -> bool:
    return call_index >= max_calls


def stream_simple_rag(
    llm_client: Any,
    query: str,
    search_context: str,
) -> Iterator[str]:
    messages = [
        {"role": "system", "content": SIMPLE_RAG_PROMPT},
        {
            "role": "user",
            "content": f"{QUESTION_LABEL}\n{query}\n\n{SEARCH_RESULTS_LABEL}\n{search_context}",
        },
    ]
    yield from llm_client.stream(messages)


def _message_from_response(response: Any) -> Any:
    return _get_value(_get_value(response, "choices")[0], "message")


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _ensure_system_prompt(state: ConversationState) -> None:
    state.messages = [
        message
        for message in state.messages
        if not (
            message.get("role") == "system"
            and message.get("content") == SYSTEM_PROMPT
        )
    ]
    state.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})


def _tool_call_id(tool_call: Any) -> str:
    return _get_value(tool_call, "id", "")


def _tool_call_function(tool_call: Any) -> Any:
    return _get_value(tool_call, "function", {})


def _tool_call_name(tool_call: Any) -> str:
    return _get_value(_tool_call_function(tool_call), "name", "")


def _parse_tool_arguments(tool_call: Any) -> dict[str, Any]:
    arguments = _get_value(_tool_call_function(tool_call), "arguments")
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments:
        return INVALID_TOOL_ARGUMENTS.copy()

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return INVALID_TOOL_ARGUMENTS.copy()

    if not isinstance(parsed, dict):
        return NON_OBJECT_TOOL_ARGUMENTS.copy()
    return parsed


def run_agentic_loop(
    llm_client: Any,
    state: ConversationState,
    tool_executor: Callable[[str, dict[str, Any]], str],
    max_calls: int,
    token_threshold: int,
    token_warning_ratio: float,
    status_writer: Callable[[str], None],
) -> Iterator[str]:
    _ensure_system_prompt(state)

    for call_index in range(1, max_calls + 1):
        state.maybe_add_token_warning(token_threshold, token_warning_ratio)
        if state.total_tokens() >= token_threshold:
            candidate_reference_ids = list(state.references.keys())
            state.summarize(candidate_reference_ids)
            tool_executor(
                "summarize",
                {"candidate_reference_ids": candidate_reference_ids},
            )

        response = llm_client.tool_call(state.messages, TOOL_SCHEMAS)
        message = _message_from_response(response)
        tool_calls = _get_value(message, "tool_calls")

        if tool_calls:
            for tool_call in tool_calls:
                name = _tool_call_name(tool_call)
                arguments = _parse_tool_arguments(tool_call)
                status_writer(f"[tool] {name}")
                result = tool_executor(name, arguments)
                state.add_message(
                    "tool",
                    result,
                    tool_call_id=_tool_call_id(tool_call),
                    name=name,
                )
            continue

        content = _get_value(message, "content") or ""
        if content:
            yield content
        return

    state.add_message("system", FORCE_FINAL_ANSWER_PROMPT)
    yield from llm_client.stream(state.messages)
