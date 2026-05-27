from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from agenticrag.models import DocumentChunk


def make_doc_id(path: Path) -> str:
    return hashlib.sha1(path.as_posix().encode("utf-8")).hexdigest()[:16]


def normalize_lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def extract_markdown_title(text: str, fallback: str) -> str:
    for line in normalize_lines(text):
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
    return fallback


def split_sentences(text: str) -> list[str]:
    parts = re.findall(r".*?[\u3002\uff01\uff1f.!?]\s*|.+$", text.strip(), flags=re.DOTALL)
    return [part for part in parts if part]


def chunk_text(
    text: str,
    max_chunk_size: int = 1000,
    overlap: int = 100,
    min_chunk_size: int = 50,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    max_chunk_size = max(1, max_chunk_size)
    overlap = max(0, overlap)
    min_chunk_size = max(1, min_chunk_size)

    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text) or [text]:
        for piece in _split_oversized_text(sentence, max_chunk_size):
            candidate = _join_chunk_parts(current, piece)
            if len(candidate) <= max_chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
                prefix = current[-overlap:] if overlap else ""
                current = _join_chunk_parts(prefix, piece)
            else:
                current = piece

    if current:
        if chunks and len(current) < min_chunk_size:
            chunks[-1] = _join_chunk_parts(chunks[-1], current)
        else:
            chunks.append(current)

    return [chunk for chunk in chunks if len(chunk) >= min_chunk_size or len(chunks) == 1]


def _split_oversized_text(text: str, max_chunk_size: int) -> list[str]:
    if len(text) <= max_chunk_size:
        return [text]
    return [text[index : index + max_chunk_size] for index in range(0, len(text), max_chunk_size)]


def _join_chunk_parts(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return f"{left}{right}"


def _line_span_for_text(lines: list[str], needle: str, start_at: int) -> tuple[int, int]:
    if not lines:
        return 0, 0

    joined = "\n".join(lines)
    search_from = min(max(start_at, 0), len(joined))
    stripped_needle = needle.strip()
    char_start = joined.find(stripped_needle, search_from)
    if char_start < 0:
        char_start = joined.find(stripped_needle)
    if char_start < 0:
        return 0, 0

    line_start = joined[:char_start].count("\n")
    line_end = line_start + stripped_needle.count("\n")
    return line_start, min(line_end, len(lines) - 1)


def parse_markdown(path: Path) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8")
    lines = normalize_lines(text)
    title = extract_markdown_title(text, path.stem)
    doc_id = make_doc_id(path)
    sections = _split_markdown_sections(text)

    chunks: list[DocumentChunk] = []
    search_start = 0
    for section in sections:
        for content in chunk_text(section):
            line_start, line_end = _line_span_for_text(lines, content, search_start)
            search_start = _advance_search_start(lines, content, line_end)
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    path=path,
                    title=title,
                    filetype="md",
                    chunk_index=len(chunks),
                    line_start=line_start,
                    line_end=line_end,
                    content=content,
                )
            )
    return chunks


def parse_pdf(path: Path) -> list[DocumentChunk]:
    import pdfplumber

    page_texts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            page_texts.append(f"[page {page_number}]\n{text}".strip())

    full_text = "\n\n".join(page_text for page_text in page_texts if page_text).strip()
    if not full_text:
        return []

    lines = normalize_lines(full_text)
    doc_id = make_doc_id(path)
    title = path.stem

    chunks: list[DocumentChunk] = []
    search_start = 0
    for content in chunk_text(full_text):
        line_start, line_end = _line_span_for_text(lines, content, search_start)
        search_start = _advance_search_start(lines, content, line_end)
        chunks.append(
            DocumentChunk(
                doc_id=doc_id,
                path=path,
                title=title,
                filetype="pdf",
                chunk_index=len(chunks),
                line_start=line_start,
                line_end=line_end,
                content=content,
            )
        )
    return chunks


def scan_documents(docs_dir: Path) -> list[Path]:
    supported_suffixes = {".md", ".pdf"}
    return sorted(
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )


def parse_document(path: Path) -> list[DocumentChunk]:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return parse_markdown(path)
    if suffix == ".pdf":
        return parse_pdf(path)
    return []


def _split_markdown_sections(text: str) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    for line in normalize_lines(text):
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [section for section in sections if section]


def _advance_search_start(lines: list[str], content: str, line_end: int) -> int:
    joined_prefix = "\n".join(lines[: line_end + 1])
    return max(len(joined_prefix) - len(content), 0)


def write_source_cache(cache_dir: Path, chunks: list[DocumentChunk]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    for chunk in chunks:
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        text = chunk.path.read_text(encoding="utf-8", errors="ignore")
        payload = {
            "doc_id": chunk.doc_id,
            "path": chunk.path.as_posix(),
            "title": chunk.title,
            "filetype": chunk.filetype,
            "lines": normalize_lines(text),
        }
        cache_path = cache_dir / f"{chunk.doc_id}.json"
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
