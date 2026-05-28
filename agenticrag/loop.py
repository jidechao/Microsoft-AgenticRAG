from __future__ import annotations

import json
import re
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
REFERENCE_ID_PATTERN = re.compile(r"\bturn\d+search\d+\b")
REFERENCE_FALLBACK_TOOL_NAMES = {"search", "find", "open"}
REFERENCE_FALLBACK_LIMIT = 5
CURRENT_TURN_TOOL_REQUIRED_PROMPT = (
    "A current-turn retrieval tool call is required before answering this complex "
    "turn. Call search, find, or open now; do not answer from prior references alone."
)
CURRENT_TURN_RETRIEVAL_FAILED_MESSAGE = (
    "证据不足：当前复杂问题没有成功的本轮检索结果，无法生成可靠回答。"
)


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


def build_reference_id_section(
    state: ConversationState,
    answer_text: str,
    *,
    tool_results_start_index: int = 0,
) -> str:
    current_reference_ids = _recent_tool_reference_ids(
        state,
        tool_results_start_index,
        limit=None,
    )
    reference_ids = _referenced_ids_in_answer(
        state,
        answer_text,
        allowed_reference_ids=set(current_reference_ids),
    )
    if not reference_ids:
        reference_ids = current_reference_ids[:REFERENCE_FALLBACK_LIMIT]
    if not reference_ids:
        return ""

    lines = ["", "", "引用标识（Reference ID）："]
    for reference_id in reference_ids:
        reference = state.references[reference_id]
        chunk = reference.chunk
        location = f"{chunk.path.as_posix()}:{chunk.line_start + 1}-{chunk.line_end + 1}"
        lines.append(f"- {reference_id}: {chunk.title} ({location})")
    return "\n".join(lines)


def _referenced_ids_in_answer(
    state: ConversationState,
    answer_text: str,
    allowed_reference_ids: set[str] | None = None,
) -> list[str]:
    reference_ids: list[str] = []
    seen: set[str] = set()
    for reference_id in REFERENCE_ID_PATTERN.findall(answer_text):
        if allowed_reference_ids is not None and reference_id not in allowed_reference_ids:
            continue
        if reference_id in seen or reference_id not in state.references:
            continue
        seen.add(reference_id)
        reference_ids.append(reference_id)
    return reference_ids


def _recent_tool_reference_ids(
    state: ConversationState,
    start_index: int,
    *,
    limit: int | None = REFERENCE_FALLBACK_LIMIT,
) -> list[str]:
    reference_ids: list[str] = []
    seen: set[str] = set()
    for tool_result in state.tool_results[start_index:]:
        if tool_result.name not in REFERENCE_FALLBACK_TOOL_NAMES:
            continue
        metadata_reference_ids = tool_result.metadata.get("reference_ids")
        if not isinstance(metadata_reference_ids, list):
            continue
        for reference_id in metadata_reference_ids:
            if not isinstance(reference_id, str):
                continue
            if reference_id in seen or reference_id not in state.references:
                continue
            seen.add(reference_id)
            reference_ids.append(reference_id)
            if limit is not None and len(reference_ids) >= limit:
                return reference_ids
    return reference_ids


def _has_current_turn_retrieval_result(
    state: ConversationState,
    start_index: int,
) -> bool:
    return bool(_recent_tool_reference_ids(state, start_index, limit=1))


def _latest_user_query(state: ConversationState) -> str:
    for message in reversed(state.messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        match = re.search(
            r"Rewritten self-contained question:\s*(.+)",
            content,
            flags=re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        if content:
            return content
    return state.user_query


def _remove_new_tool_messages(state: ConversationState, start_index: int) -> None:
    state.messages = [
        *state.messages[:start_index],
        *[
            message
            for message in state.messages[start_index:]
            if message.get("role") != "tool"
        ],
    ]


def _run_automatic_search(
    state: ConversationState,
    tool_executor: Callable[[str, dict[str, Any]], str],
    status_writer: Callable[[str], None],
) -> str:
    status_writer("[tool] search")
    message_start_index = len(state.messages)
    result = tool_executor("search", {"queries": [_latest_user_query(state)]})
    _remove_new_tool_messages(state, message_start_index)
    if result.startswith("[tool error]"):
        status_writer(result)
    else:
        state.add_message(
            "system",
            "Current-turn automatic search result:\n" + result,
        )
    return result


def _has_multi_step_current_turn_tool_activity(
    state: ConversationState,
    start_index: int,
) -> bool:
    return len(state.tool_results[start_index:]) >= 2


def _format_tool_results_for_final_answer(
    state: ConversationState,
    start_index: int,
) -> str:
    lines: list[str] = []
    for tool_result in state.tool_results[start_index:]:
        lines.append(f"[{tool_result.name}]")
        lines.append(tool_result.content)
    return "\n".join(lines).strip()


def _build_streaming_final_messages(
    state: ConversationState,
    tool_results_start_index: int,
    *,
    force_final: bool = False,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in state.messages:
        if message.get("role") == "system":
            continue
        if message.get("role") == "tool":
            continue
        if message.get("tool_calls"):
            continue
        content = message.get("content")
        if content is None:
            continue
        messages.append({"role": message.get("role", "user"), "content": content})

    evidence = _format_tool_results_for_final_answer(state, tool_results_start_index)
    messages.insert(
        0,
        {
            "role": "system",
            "content": (
                "You are writing the final user-facing answer after retrieval is complete. "
                + (
                    "Tool budget is exhausted. "
                    if force_final
                    else ""
                )
                + "Do not call tools. Do not emit tool-call markup, XML tags, DSML traces, "
                "or any protocol/internal formatting. Answer in plain Chinese prose and cite "
                "Reference IDs inline when useful."
            ),
        },
    )
    if evidence:
        messages.append(
            {
                "role": "system",
                "content": f"Collected tool evidence:\n{evidence}",
            }
        )
    return messages


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


def _tool_call_as_message_dict(tool_call: Any) -> dict[str, Any]:
    function = _tool_call_function(tool_call)
    return {
        "id": _tool_call_id(tool_call),
        "type": _get_value(tool_call, "type", "function"),
        "function": {
            "name": _get_value(function, "name", ""),
            "arguments": _get_value(function, "arguments", ""),
        },
    }


def _attach_tool_call_id_to_new_tool_message(
    state: ConversationState,
    start_index: int,
    tool_call_id: str,
    name: str,
) -> bool:
    for message in reversed(state.messages[start_index:]):
        if message.get("role") != "tool":
            continue
        message["tool_call_id"] = tool_call_id
        message["name"] = name
        return True
    return False


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


def run_agentic_loop(
    llm_client: Any,
    state: ConversationState,
    tool_executor: Callable[[str, dict[str, Any]], str],
    max_calls: int,
    token_threshold: int,
    token_warning_ratio: float,
    status_writer: Callable[[str], None],
    require_current_turn_retrieval: bool = False,
) -> Iterator[str]:
    _ensure_system_prompt(state)
    tool_results_start_index = len(state.tool_results)
    prompted_for_current_turn_tool = False
    automatic_search_attempted = False

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
            state.add_message(
                "assistant",
                _get_value(message, "content") or "",
                tool_calls=[
                    _tool_call_as_message_dict(tool_call) for tool_call in tool_calls
                ],
            )
            for tool_call in tool_calls:
                name = _tool_call_name(tool_call)
                arguments = _parse_tool_arguments(tool_call)
                status_writer(f"[tool] {name}")
                message_start_index = len(state.messages)
                result = tool_executor(name, arguments)
                if result.startswith("[tool error]"):
                    status_writer(result)
                tool_call_id = _tool_call_id(tool_call)
                if not _attach_tool_call_id_to_new_tool_message(
                    state,
                    message_start_index,
                    tool_call_id,
                    name,
                ):
                    state.add_message(
                        "tool",
                        result,
                        tool_call_id=tool_call_id,
                        name=name,
                    )
            continue

        content = _get_value(message, "content") or ""
        if content:
            if require_current_turn_retrieval and not _has_current_turn_retrieval_result(
                state,
                tool_results_start_index,
            ):
                if not automatic_search_attempted:
                    _run_automatic_search(state, tool_executor, status_writer)
                    automatic_search_attempted = True
                    if _has_current_turn_retrieval_result(
                        state,
                        tool_results_start_index,
                    ):
                        continue
                state.add_message("system", CURRENT_TURN_TOOL_REQUIRED_PROMPT)
                prompted_for_current_turn_tool = True
                continue
            if _has_multi_step_current_turn_tool_activity(state, tool_results_start_index):
                yield from llm_client.stream(
                    _build_streaming_final_messages(state, tool_results_start_index)
                )
                return
            yield content
        return

    if (
        not require_current_turn_retrieval
        or _has_current_turn_retrieval_result(state, tool_results_start_index)
    ):
        state.add_message("system", FORCE_FINAL_ANSWER_PROMPT)
        yield from llm_client.stream(
            _build_streaming_final_messages(
                state,
                tool_results_start_index,
                force_final=True,
            )
        )
        return

    yield CURRENT_TURN_RETRIEVAL_FAILED_MESSAGE


def run_ask(query: str) -> int:
    from agenticrag.config import load_config
    from agenticrag.embeddings import SiliconFlowEmbeddingClient
    from agenticrag.llm import DeepSeekClient
    from agenticrag.retriever import ChromaRetriever
    from agenticrag.state import ConversationState
    from agenticrag.switcher import classify_query
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
    state = ConversationState(user_query=query)
    tools = RetrievalTools(
        retriever=retriever,
        state=state,
        source_cache_dir=config.source_cache_dir,
    )

    route = classify_query(llm_client, query)
    if route == "simple":
        tool_results_start_index = len(state.tool_results)
        context = tools.search([query])
        chunks: list[str] = []
        for chunk in stream_simple_rag(llm_client, query, context):
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        reference_section = build_reference_id_section(
            state,
            "".join(chunks),
            tool_results_start_index=tool_results_start_index,
        )
        if reference_section:
            print(reference_section, end="", flush=True)
        print()
        return 0

    def execute_tool(name: str, arguments: dict[str, Any]) -> str:
        return execute_retrieval_tool(tools, name, arguments)

    tool_results_start_index = len(state.tool_results)
    chunks = []
    for chunk in run_agentic_loop(
        llm_client,
        state,
        execute_tool,
        max_calls=config.max_calls,
        token_threshold=config.token_threshold,
        token_warning_ratio=config.token_warning_ratio,
        status_writer=lambda text: print(text, flush=True),
        require_current_turn_retrieval=True,
    ):
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    reference_section = build_reference_id_section(
        state,
        "".join(chunks),
        tool_results_start_index=tool_results_start_index,
    )
    if reference_section:
        print(reference_section, end="", flush=True)
    print()
    return 0
