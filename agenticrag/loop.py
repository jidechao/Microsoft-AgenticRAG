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
    return response.choices[0].message


def run_agentic_loop(
    llm_client: Any,
    state: ConversationState,
    tool_executor: Callable[[str, dict[str, Any]], str],
    max_calls: int,
    token_threshold: int,
    token_warning_ratio: float,
    status_writer: Callable[[str], None],
) -> Iterator[str]:
    state.add_message("system", SYSTEM_PROMPT)

    for call_index in range(1, max_calls + 1):
        state.maybe_add_token_warning(token_threshold, token_warning_ratio)
        if state.total_tokens() >= token_threshold:
            tool_executor(
                "summarize",
                {"candidate_reference_ids": list(state.references.keys())},
            )

        response = llm_client.tool_call(state.messages, TOOL_SCHEMAS)
        message = _message_from_response(response)
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                status_writer(f"[tool] {name}")
                result = tool_executor(name, arguments)
                state.add_message(
                    "tool",
                    result,
                    tool_call_id=tool_call.id,
                    name=name,
                )
            continue

        content = getattr(message, "content", None) or ""
        if content:
            yield content
        return

    state.add_message("system", FORCE_FINAL_ANSWER_PROMPT)
    yield from llm_client.stream(state.messages)
