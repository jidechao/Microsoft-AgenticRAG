from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from agenticrag.models import DocumentChunk, RetrievedChunk
from agenticrag.state import ConversationState


class Retriever(Protocol):
    def query(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        ...


class RetrievalTools:
    def __init__(
        self,
        *,
        retriever: Retriever,
        state: ConversationState,
        source_cache_dir: Path | str,
    ) -> None:
        self.retriever = retriever
        self.state = state
        self.source_cache_dir = Path(source_cache_dir)

    def _load_source(self, doc_id: str) -> dict[str, Any]:
        cache_path = self.source_cache_dir / f"{doc_id}.json"
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing source cache for doc_id: {doc_id}") from exc
        if not isinstance(payload.get("lines"), list):
            raise ValueError(f"Invalid source cache for doc_id: {doc_id}")
        return payload

    def search(self, queries: list[str]) -> str:
        chunks: list[DocumentChunk] = []
        seen: set[tuple[str, str]] = set()
        for query in queries[:5]:
            for result in self.retriever.query(query, top_k=10):
                key = (result.chunk.doc_id, result.chunk.content)
                if key in seen:
                    continue
                seen.add(key)
                chunks.append(result.chunk)
                if len(chunks) >= 10:
                    break
            if len(chunks) >= 10:
                break

        reference_ids = self.state.assign_search_results(chunks)
        lines = [
            self._format_search_block(reference_id, chunk)
            for reference_id, chunk in zip(reference_ids, chunks, strict=True)
        ]
        content = "\n\n".join(lines) if lines else "No search results."
        self.state.add_tool_result(
            "search",
            content,
            metadata={"reference_ids": reference_ids},
        )
        return content

    def find(self, reference_id: str, patterns: list[str]) -> str:
        reference = self.state.get_reference(reference_id)
        source = self._load_source(reference.chunk.doc_id)
        lines = [str(line) for line in source["lines"]]
        blocks: list[str] = []

        for pattern in patterns:
            pattern_lower = pattern.lower()
            matches: list[str] = []
            if pattern_lower:
                for line_index, line in enumerate(lines):
                    match_index = line.lower().find(pattern_lower)
                    if match_index == -1:
                        continue
                    start = max(match_index - 50, 0)
                    end = min(match_index + len(pattern) + 50, len(line))
                    context = line[start:end]
                    matches.append(f"line {line_index + 1}: {context}")
                    if len(matches) >= 2:
                        break
            if matches:
                blocks.append(f"Pattern {pattern!r}:\n" + "\n".join(matches))
            else:
                blocks.append(f"Pattern {pattern!r}: no matches")

        content = "\n\n".join(blocks)
        self.state.add_tool_result(
            "find",
            content,
            metadata={"reference_ids": [reference_id]},
        )
        return content

    def open(self, reference_id: str, line_number: int = 0) -> str:
        reference = self.state.get_reference(reference_id)
        source = self._load_source(reference.chunk.doc_id)
        lines = [str(line) for line in source["lines"]]
        total_lines = len(lines)
        center = line_number or reference.chunk.line_start or 1
        center = min(max(center, 1), max(total_lines, 1))
        start = max(center - 20, 1)
        end = min(center + 20, total_lines)
        numbered = [
            f"{line_index}: {lines[line_index - 1]}"
            for line_index in range(start, end + 1)
        ]
        content = (
            f"Viewing lines [{start}-{end}] of {total_lines} total lines"
            + ("\n" + "\n".join(numbered) if numbered else "")
        )
        self.state.add_tool_result(
            "open",
            content,
            metadata={"reference_ids": [reference_id]},
        )
        return content

    def summarize(self, candidate_reference_ids: list[str]) -> str:
        self.state.summarize(candidate_reference_ids)
        content = "Summarized prior tool results."
        self.state.add_tool_result(
            "summarize",
            content,
            metadata={"reference_ids": list(candidate_reference_ids)},
        )
        return content

    @staticmethod
    def _format_search_block(reference_id: str, chunk: DocumentChunk) -> str:
        location = f"{chunk.path.as_posix()}:{chunk.line_start}-{chunk.line_end}"
        return (
            f"[{reference_id}] {chunk.title} ({location})\n"
            f"{chunk.content}"
        )
