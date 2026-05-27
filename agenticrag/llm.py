from __future__ import annotations

from typing import Any, Iterable, Iterator

from openai import OpenAI


def collect_stream_text(events: Iterable[Any]) -> Iterator[str]:
    for event in events:
        choices = getattr(event, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None)
        if content:
            yield content


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def complete(self, messages: list[dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
        )
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        return getattr(message, "content", None) or ""

    def stream(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        events = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        yield from collect_stream_text(events)

    def tool_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=False,
        )
