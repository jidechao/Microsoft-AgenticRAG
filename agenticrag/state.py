from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tiktoken

from agenticrag.models import DocumentChunk, Reference, ToolResult


@dataclass
class ConversationState:
    user_query: str
    turn_index: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    references: dict[str, Reference] = field(default_factory=dict)
    tool_results: list[ToolResult] = field(default_factory=list)
    warned_about_tokens: bool = False

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages.append({"role": "user", "content": self.user_query})

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        message = {"role": role, "content": content}
        message.update(extra)
        self.messages.append(message)

    def assign_search_results(self, chunks: list[DocumentChunk]) -> list[str]:
        reference_ids: list[str] = []
        for index, chunk in enumerate(chunks):
            reference_id = f"turn{self.turn_index}search{index}"
            self.references[reference_id] = Reference(
                reference_id=reference_id,
                chunk=chunk,
            )
            reference_ids.append(reference_id)
        return reference_ids

    def get_reference(self, reference_id: str) -> Reference:
        try:
            return self.references[reference_id]
        except KeyError as exc:
            raise KeyError(f"Unknown reference_id: {reference_id}") from exc

    def add_tool_result(
        self,
        name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        tool_result = ToolResult(name=name, content=content, metadata=dict(metadata or {}))
        self.tool_results.append(tool_result)
        self.messages.append({"role": "tool", "name": name, "content": content})

    def total_tokens(self) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for message in self.messages:
            total += len(encoding.encode(str(message.get("content", ""))))
        return total

    def maybe_add_token_warning(self, threshold: int, ratio: float) -> bool:
        if self.warned_about_tokens:
            return False
        if self.total_tokens() >= int(threshold * ratio):
            self.add_message(
                "system",
                "Internal warning: token usage is near the context limit. Prefer summarize before opening more long documents.",
            )
            self.warned_about_tokens = True
            return True
        return False

    def summarize(self, candidate_reference_ids: list[str]) -> None:
        retained = set(candidate_reference_ids)
        for tool_result in self.tool_results:
            reference_ids = set(tool_result.metadata.get("reference_ids", []))
            if reference_ids and reference_ids.isdisjoint(retained):
                tool_result.content = (
                    f"[compressed {tool_result.name} result unrelated to retained references]"
                )

        non_tool_messages = [
            message for message in self.messages if message.get("role") != "tool"
        ]
        self.messages = non_tool_messages
        for tool_result in self.tool_results:
            self.messages.append(
                {"role": "tool", "name": tool_result.name, "content": tool_result.content}
            )
