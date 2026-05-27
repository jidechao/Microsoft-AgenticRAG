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
            line_number = arguments["line_number"]
            if not isinstance(reference_id, str):
                return "[tool error] open: reference_id must be a string"
            if not isinstance(line_number, int):
                raise ValueError("line_number must be an integer")
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
            yield content
        return

    state.add_message("system", FORCE_FINAL_ANSWER_PROMPT)
    yield from llm_client.stream(state.messages)


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
        context = tools.search([query])
        for chunk in stream_simple_rag(llm_client, query, context):
            print(chunk, end="", flush=True)
        print()
        return 0

    def execute_tool(name: str, arguments: dict[str, Any]) -> str:
        return execute_retrieval_tool(tools, name, arguments)

    for chunk in run_agentic_loop(
        llm_client,
        state,
        execute_tool,
        max_calls=config.max_calls,
        token_threshold=config.token_threshold,
        token_warning_ratio=config.token_warning_ratio,
        status_writer=lambda text: print(text, flush=True),
    ):
        print(chunk, end="", flush=True)
    print()
    return 0
