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
        required_fields = {
            "doc_id": str,
            "path": str,
            "title": str,
            "filetype": str,
            "lines": list,
        }
        for field_name, field_type in required_fields.items():
            if not isinstance(payload.get(field_name), field_type):
                raise ValueError(
                    f"Invalid source cache for doc_id: {doc_id}; "
                    f"missing or invalid {field_name}"
                )
        if payload["doc_id"] != doc_id:
            raise ValueError(
                f"Invalid source cache for doc_id: {doc_id}; "
                f"cached doc_id is {payload['doc_id']!r}"
            )
        if not all(isinstance(line, str) for line in payload["lines"]):
            raise ValueError(f"Invalid source cache for doc_id: {doc_id}")
        return payload

    def search(self, queries: list[str]) -> str:
        chunks: list[DocumentChunk] = []
        seen_chunks: set[tuple[str, int, int, int, str]] = set()
        normalized_queries = self._normalize_values(queries, limit=5)
        for query in normalized_queries:
            for result in self.retriever.query(query, top_k=10):
                key = (
                    result.chunk.doc_id,
                    result.chunk.chunk_index,
                    result.chunk.line_start,
                    result.chunk.line_end,
                    result.chunk.content,
                )
                if key in seen_chunks:
                    continue
                seen_chunks.add(key)
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
        reference = self._get_reference_for_tool("find", reference_id)
        if reference is None:
            return self._store_error("find", f"Unknown reference_id: {reference_id}", [reference_id])
        source = self._load_source_for_tool("find", reference.chunk, reference_id)
        if source is None:
            return self.state.tool_results[-1].content
        lines = [str(line) for line in source["lines"]]
        blocks: list[str] = []

        normalized_patterns = self._normalize_values(patterns, limit=10)
        if not normalized_patterns:
            content = "No non-empty patterns provided."
            self.state.add_tool_result(
                "find",
                content,
                metadata={"reference_ids": [reference_id]},
            )
            return content

        for pattern in normalized_patterns:
            pattern_lower = pattern.lower()
            matches: list[str] = []
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
        reference = self._get_reference_for_tool("open", reference_id)
        if reference is None:
            return self._store_error("open", f"Unknown reference_id: {reference_id}", [reference_id])
        source = self._load_source_for_tool("open", reference.chunk, reference_id)
        if source is None:
            return self.state.tool_results[-1].content
        lines = [str(line) for line in source["lines"]]
        total_lines = len(lines)
        center = line_number if line_number > 0 else reference.chunk.line_start + 1
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
        retained_reference_ids = [
            reference_id
            for reference_id in self._normalize_values(candidate_reference_ids)
            if reference_id in self.state.references
        ]
        unknown_reference_ids = [
            reference_id
            for reference_id in self._normalize_values(candidate_reference_ids)
            if reference_id not in self.state.references
        ]
        if retained_reference_ids:
            self.state.summarize(retained_reference_ids)
        if unknown_reference_ids and not retained_reference_ids:
            content = (
                "[tool error] summarize: unknown reference_id(s): "
                + ", ".join(unknown_reference_ids)
            )
        elif unknown_reference_ids:
            content = (
                "Summarized prior tool results. Ignored unknown reference_id(s): "
                + ", ".join(unknown_reference_ids)
            )
        else:
            content = "Summarized prior tool results."
        self.state.add_tool_result(
            "summarize",
            content,
            metadata={
                "reference_ids": retained_reference_ids,
                "unknown_reference_ids": unknown_reference_ids,
            },
        )
        return content

    @staticmethod
    def _format_search_block(reference_id: str, chunk: DocumentChunk) -> str:
        location = (
            f"{chunk.path.as_posix()}:"
            f"{chunk.line_start + 1}-{chunk.line_end + 1}"
        )
        return (
            f"[{reference_id}] {chunk.title} ({location})\n"
            f"{chunk.content}"
        )

    @staticmethod
    def _normalize_values(values: list[str], limit: int | None = None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = value.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
            if limit is not None and len(normalized) >= limit:
                break
        return normalized

    def _get_reference_for_tool(self, tool_name: str, reference_id: str):
        try:
            return self.state.get_reference(reference_id)
        except KeyError:
            return None

    def _load_source_for_tool(
        self,
        tool_name: str,
        chunk: DocumentChunk,
        reference_id: str,
    ) -> dict[str, Any] | None:
        try:
            source = self._load_source(chunk.doc_id)
            self._validate_source_for_chunk(source, chunk)
            return source
        except (FileNotFoundError, ValueError) as exc:
            self._store_error(tool_name, str(exc), [reference_id])
            return None

    @staticmethod
    def _validate_source_for_chunk(source: dict[str, Any], chunk: DocumentChunk) -> None:
        expected_path = chunk.path.as_posix()
        if source["doc_id"] != chunk.doc_id:
            raise ValueError(
                f"Source cache mismatch for doc_id {chunk.doc_id}: "
                f"cached doc_id is {source['doc_id']!r}"
            )
        if source["path"] != expected_path:
            raise ValueError(
                f"Source cache mismatch for doc_id {chunk.doc_id}: "
                f"cached path is {source['path']!r}, expected {expected_path!r}"
            )

    def _store_error(
        self,
        tool_name: str,
        message: str,
        reference_ids: list[str] | None = None,
    ) -> str:
        content = f"[tool error] {tool_name}: {message}"
        self.state.add_tool_result(
            tool_name,
            content,
            metadata={"reference_ids": list(reference_ids or [])},
        )
        return content
