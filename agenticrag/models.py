from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    doc_id: str
    path: Path
    title: str
    filetype: str
    chunk_index: int
    line_start: int
    line_end: int
    content: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "path": self.path.as_posix(),
            "title": self.title,
            "filetype": self.filetype,
            "chunk_index": self.chunk_index,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float

    @property
    def snippet(self) -> str:
        return self.chunk.content[:200]


@dataclass
class Reference:
    reference_id: str
    chunk: DocumentChunk


@dataclass
class ToolResult:
    name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
