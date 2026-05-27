from __future__ import annotations

import json
import re
from typing import Any, Literal

from agenticrag.prompts import SWITCHER_PROMPT

Route = Literal["simple", "complex"]


def parse_switcher_decision(text: str) -> Route:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return "complex"

    try:
        payload: Any = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "complex"

    if not isinstance(payload, dict):
        return "complex"

    return "simple" if payload.get("route") == "simple" else "complex"


def classify_query(llm_client, query: str) -> Route:
    response = llm_client.complete(
        messages=[
            {"role": "system", "content": SWITCHER_PROMPT},
            {"role": "user", "content": query},
        ]
    )
    return parse_switcher_decision(response)
