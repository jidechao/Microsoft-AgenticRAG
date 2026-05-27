from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

from agenticrag.prompts import SWITCHER_PROMPT

Route = Literal["simple", "complex"]
Message = dict[str, str]


class SupportsComplete(Protocol):
    def complete(self, *, messages: list[Message]) -> str: ...


def _json_candidates(text: str) -> list[str]:
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.extend(re.findall(r"\{.*?\}", text, flags=re.DOTALL))
    return candidates


def parse_switcher_decision(text: str) -> Route:
    for candidate in _json_candidates(text):
        try:
            payload: Any = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue

        return "simple" if payload.get("route") == "simple" else "complex"

    return "complex"


def classify_query(llm_client: SupportsComplete, query: str) -> Route:
    response = llm_client.complete(
        messages=[
            {"role": "system", "content": SWITCHER_PROMPT},
            {"role": "user", "content": query},
        ]
    )
    return parse_switcher_decision(response)
