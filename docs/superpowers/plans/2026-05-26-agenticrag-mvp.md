# AgenticRAG MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI MVP that indexes local Markdown/PDF documents and answers questions with traditional RAG or AgenticRAG, using streaming final output.

**Architecture:** The system is a small Python application with a CLI entry point, a local Chroma index, provider clients for DeepSeek and SiliconFlow, and focused modules for ingestion, retrieval, state, tools, routing, and the agentic loop. The implementation favors testable pure functions first, then wires external APIs at the edges.

**Tech Stack:** Python 3.10+, pytest, python-dotenv, openai-compatible clients, chromadb, pdfplumber, tiktoken, SiliconFlow Qwen3-Embedding-4B, DeepSeek chat/completions API.

---

## File Structure

- Create `requirements.txt`: runtime and test dependencies.
- Create `.env.example`: documented local configuration template.
- Modify `.gitignore`: keep `.env`, Chroma data, caches, and build artifacts out of Git.
- Create `agenticrag/__init__.py`: package marker.
- Create `agenticrag/config.py`: environment loading, typed settings, and validation.
- Create `agenticrag/models.py`: shared dataclasses for chunks, retrieval results, tool results, and tool calls.
- Create `agenticrag/ingest.py`: Markdown/PDF parsing, line normalization, chunking, and source cache writing.
- Create `agenticrag/embeddings.py`: SiliconFlow embedding client and deterministic fake embedding for tests.
- Create `agenticrag/retriever.py`: Chroma collection wrapper with add/query operations.
- Create `agenticrag/state.py`: conversation messages, Reference ID mapping, token estimates, and compression.
- Create `agenticrag/tools/__init__.py`: exports for tool schemas and executor.
- Create `agenticrag/tools/schemas.py`: OpenAI-compatible tool definitions.
- Create `agenticrag/tools/retrieval.py`: `search`, `find`, `open`, and `summarize` implementations.
- Create `agenticrag/llm.py`: DeepSeek client adapter for normal, streaming, and tool-calling responses.
- Create `agenticrag/prompts.py`: system, simple RAG, switcher, and forced-completion prompts.
- Create `agenticrag/switcher.py`: simple/complex query classification.
- Create `agenticrag/loop.py`: traditional RAG and Agentic Loop orchestration.
- Create `main.py`: CLI commands `index` and `ask`.
- Create `tests/`: focused pytest coverage for configuration, chunking, state, tools, switcher parsing, and loop control.

---

### Task 1: Project Skeleton And Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Modify: `.gitignore`
- Create: `agenticrag/__init__.py`
- Create: `tests/test_imports.py`

- [ ] **Step 1: Write the import smoke test**

Create `tests/test_imports.py`:

```python
def test_package_imports():
    import agenticrag

    assert agenticrag.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
pytest tests/test_imports.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agenticrag'`.

- [ ] **Step 3: Add dependencies and package marker**

Create `requirements.txt`:

```text
chromadb>=0.5.0
openai>=1.40.0
python-dotenv>=1.0.1
pdfplumber>=0.11.0
tiktoken>=0.7.0
pytest>=8.2.0
```

Create `agenticrag/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `.env.example`:

```text
DEEPSEEK_API_KEY=
SILICONFLOW_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
EMBEDDING_DIMS=1536
DOCS_DIR=docs
CHROMA_DIR=.chroma
SOURCE_CACHE_DIR=.agenticrag_cache
MAX_CALLS=15
TOKEN_THRESHOLD=128000
TOKEN_WARNING_RATIO=0.9
```

Ensure `.gitignore` contains:

```text
.env
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.chroma/
chroma/
.agenticrag_cache/
*.log
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```powershell
pytest tests/test_imports.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add .gitignore .env.example requirements.txt agenticrag/__init__.py tests/test_imports.py
git commit -m "chore: add Python project skeleton"
```

---

### Task 2: Configuration Loading

**Files:**
- Create: `agenticrag/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from agenticrag.config import Config, ConfigError, load_config


def test_load_config_uses_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "silicon-key")

    config = load_config(load_dotenv_file=False)

    assert config.deepseek_api_key == "deepseek-key"
    assert config.siliconflow_api_key == "silicon-key"
    assert config.deepseek_base_url == "https://api.deepseek.com"
    assert config.deepseek_model == "deepseek-chat"
    assert config.siliconflow_embedding_model == "Qwen/Qwen3-Embedding-4B"
    assert config.embedding_dims == 1536
    assert config.docs_dir == Path("docs")
    assert config.chroma_dir == Path(".chroma")
    assert config.source_cache_dir == Path(".agenticrag_cache")
    assert config.max_calls == 15
    assert config.token_threshold == 128000
    assert config.token_warning_ratio == 0.9


def test_load_config_requires_api_keys(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

    with pytest.raises(ConfigError) as exc:
        load_config(load_dotenv_file=False)

    assert "DEEPSEEK_API_KEY" in str(exc.value)
    assert "SILICONFLOW_API_KEY" in str(exc.value)


def test_config_validates_warning_ratio():
    with pytest.raises(ConfigError, match="TOKEN_WARNING_RATIO"):
        Config(
            deepseek_api_key="a",
            siliconflow_api_key="b",
            token_warning_ratio=1.5,
        ).validate()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agenticrag.config'`.

- [ ] **Step 3: Implement configuration module**

Create `agenticrag/config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(ValueError):
    pass


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float") from exc


@dataclass
class Config:
    deepseek_api_key: str
    siliconflow_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_dims: int = 1536
    docs_dir: Path = Path("docs")
    chroma_dir: Path = Path(".chroma")
    source_cache_dir: Path = Path(".agenticrag_cache")
    max_calls: int = 15
    token_threshold: int = 128000
    token_warning_ratio: float = 0.9

    def validate(self) -> "Config":
        missing = []
        if not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if not self.siliconflow_api_key:
            missing.append("SILICONFLOW_API_KEY")
        if missing:
            raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")
        if self.embedding_dims <= 0:
            raise ConfigError("EMBEDDING_DIMS must be positive")
        if self.max_calls <= 0:
            raise ConfigError("MAX_CALLS must be positive")
        if self.token_threshold <= 0:
            raise ConfigError("TOKEN_THRESHOLD must be positive")
        if not 0 < self.token_warning_ratio < 1:
            raise ConfigError("TOKEN_WARNING_RATIO must be between 0 and 1")
        return self


def load_config(load_dotenv_file: bool = True) -> Config:
    if load_dotenv_file:
        load_dotenv()

    config = Config(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        siliconflow_api_key=os.getenv("SILICONFLOW_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        siliconflow_base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        siliconflow_embedding_model=os.getenv(
            "SILICONFLOW_EMBEDDING_MODEL",
            "Qwen/Qwen3-Embedding-4B",
        ),
        embedding_dims=_int_env("EMBEDDING_DIMS", 1536),
        docs_dir=Path(os.getenv("DOCS_DIR", "docs")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", ".chroma")),
        source_cache_dir=Path(os.getenv("SOURCE_CACHE_DIR", ".agenticrag_cache")),
        max_calls=_int_env("MAX_CALLS", 15),
        token_threshold=_int_env("TOKEN_THRESHOLD", 128000),
        token_warning_ratio=_float_env("TOKEN_WARNING_RATIO", 0.9),
    )
    return config.validate()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add agenticrag/config.py tests/test_config.py
git commit -m "feat: load AgenticRAG configuration"
```

---

### Task 3: Shared Models

**Files:**
- Create: `agenticrag/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_models.py`:

```python
from pathlib import Path

from agenticrag.models import DocumentChunk, RetrievedChunk


def test_document_chunk_metadata_round_trip():
    chunk = DocumentChunk(
        doc_id="abc",
        path=Path("docs/a.md"),
        title="Title",
        filetype="md",
        chunk_index=2,
        line_start=10,
        line_end=20,
        content="hello",
    )

    metadata = chunk.to_metadata()

    assert metadata["doc_id"] == "abc"
    assert metadata["path"] == "docs/a.md"
    assert metadata["title"] == "Title"
    assert metadata["filetype"] == "md"
    assert metadata["chunk_index"] == 2
    assert metadata["line_start"] == 10
    assert metadata["line_end"] == 20


def test_retrieved_chunk_builds_snippet():
    chunk = DocumentChunk(
        doc_id="abc",
        path=Path("docs/a.md"),
        title="Title",
        filetype="md",
        chunk_index=0,
        line_start=0,
        line_end=1,
        content="a" * 250,
    )

    result = RetrievedChunk(chunk=chunk, score=0.42)

    assert result.snippet == ("a" * 200)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_models.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agenticrag.models'`.

- [ ] **Step 3: Implement shared dataclasses**

Create `agenticrag/models.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
pytest tests/test_models.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add agenticrag/models.py tests/test_models.py
git commit -m "feat: add shared AgenticRAG models"
```

---

### Task 4: Markdown Parsing And Chunking

**Files:**
- Create: `agenticrag/ingest.py`
- Test: `tests/test_ingest_markdown.py`

- [ ] **Step 1: Write failing Markdown ingestion tests**

Create `tests/test_ingest_markdown.py`:

```python
from pathlib import Path

from agenticrag.ingest import chunk_text, parse_markdown


def test_chunk_text_respects_size_and_overlap():
    chunks = chunk_text(
        "第一句。" * 200,
        max_chunk_size=100,
        overlap=10,
        min_chunk_size=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 110 for chunk in chunks)
    assert all(len(chunk) >= 20 for chunk in chunks)


def test_parse_markdown_uses_heading_title_and_lines(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text("# Main\n\nIntro\n\n## Section\n\nBody line\n", encoding="utf-8")

    chunks = parse_markdown(path)

    assert chunks
    assert chunks[0].title == "Main"
    assert chunks[0].filetype == "md"
    assert chunks[0].path == path
    assert chunks[0].line_start >= 0
    assert chunks[0].line_end >= chunks[0].line_start
    assert "Intro" in " ".join(chunk.content for chunk in chunks)
    assert "Body line" in " ".join(chunk.content for chunk in chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_ingest_markdown.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing functions.

- [ ] **Step 3: Implement Markdown parsing and chunking**

Create `agenticrag/ingest.py`:

```python
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
    parts = re.split(r"(?<=[。！？.!?])\s*", text.strip())
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
    sentences = split_sentences(text)
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current}{sentence}" if not current else f"{current} {sentence}"
        if len(candidate) <= max_chunk_size:
            current = candidate
            continue
        if len(current) >= min_chunk_size:
            chunks.append(current)
            current = current[-overlap:] + " " + sentence if overlap > 0 else sentence
        else:
            chunks.append(candidate[:max_chunk_size])
            current = candidate[max_chunk_size - overlap :] if overlap > 0 else candidate[max_chunk_size:]
    if len(current.strip()) >= min_chunk_size or not chunks:
        chunks.append(current.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _line_span_for_text(lines: list[str], needle: str, start_at: int) -> tuple[int, int]:
    if not lines:
        return 0, 0
    joined = "\n".join(lines)
    char_start = joined.find(needle.strip(), max(0, start_at))
    if char_start < 0:
        return 0, min(len(lines) - 1, 0)
    before = joined[:char_start]
    line_start = before.count("\n")
    line_end = line_start + needle.count("\n")
    return line_start, min(line_end, len(lines) - 1)


def parse_markdown(path: Path) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8")
    lines = normalize_lines(text)
    title = extract_markdown_title(text, path.stem)
    doc_id = make_doc_id(path)

    sections = re.split(r"(?m)^##\s+", text)
    chunks: list[DocumentChunk] = []
    search_start = 0
    for section in sections:
        for content in chunk_text(section):
            line_start, line_end = _line_span_for_text(lines, content, search_start)
            search_start += max(len(content), 1)
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
        (cache_dir / f"{chunk.doc_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
pytest tests/test_ingest_markdown.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add agenticrag/ingest.py tests/test_ingest_markdown.py
git commit -m "feat: parse and chunk markdown documents"
```

---

### Task 5: PDF Parsing And Document Scanning

**Files:**
- Modify: `agenticrag/ingest.py`
- Test: `tests/test_ingest_scan.py`

- [ ] **Step 1: Write failing scan tests**

Create `tests/test_ingest_scan.py`:

```python
from pathlib import Path

from agenticrag.ingest import parse_document, scan_documents


def test_scan_documents_finds_supported_files(tmp_path):
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("skip", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.md").write_text("# C\n", encoding="utf-8")

    paths = scan_documents(tmp_path)

    assert paths == [tmp_path / "a.md", tmp_path / "nested" / "c.md"]


def test_parse_document_rejects_unsupported_file(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("skip", encoding="utf-8")

    chunks = parse_document(path)

    assert chunks == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_ingest_scan.py -v
```

Expected: FAIL with missing `scan_documents` or `parse_document`.

- [ ] **Step 3: Add PDF parsing and scanning**

Append to `agenticrag/ingest.py`:

```python
def parse_pdf(path: Path) -> list[DocumentChunk]:
    import pdfplumber

    doc_id = make_doc_id(path)
    title = path.stem
    page_texts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                page_texts.append(f"[page {page_number}]\n{text}")

    full_text = "\n\n".join(page_texts)
    lines = normalize_lines(full_text)
    chunks: list[DocumentChunk] = []
    search_start = 0
    for content in chunk_text(full_text):
        line_start, line_end = _line_span_for_text(lines, content, search_start)
        search_start += max(len(content), 1)
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
    supported = {".md", ".pdf"}
    return sorted(
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    )


def parse_document(path: Path) -> list[DocumentChunk]:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return parse_markdown(path)
    if suffix == ".pdf":
        return parse_pdf(path)
    return []
```

- [ ] **Step 4: Run ingestion tests**

Run:

```powershell
pytest tests/test_ingest_markdown.py tests/test_ingest_scan.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add agenticrag/ingest.py tests/test_ingest_scan.py
git commit -m "feat: scan and parse supported documents"
```

---

### Task 6: Embedding Client

**Files:**
- Create: `agenticrag/embeddings.py`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing embedding tests**

Create `tests/test_embeddings.py`:

```python
from agenticrag.embeddings import FakeEmbeddingClient


def test_fake_embedding_is_deterministic():
    client = FakeEmbeddingClient(dims=4)

    first = client.embed_texts(["alpha", "beta"])
    second = client.embed_texts(["alpha", "beta"])

    assert first == second
    assert len(first) == 2
    assert len(first[0]) == 4
    assert first[0] != first[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests/test_embeddings.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement embedding clients**

Create `agenticrag/embeddings.py`:

```python
from __future__ import annotations

import hashlib
import random
from typing import Protocol

from openai import OpenAI


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        pass


class FakeEmbeddingClient:
    def __init__(self, dims: int = 16) -> None:
        self.dims = dims

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)
            rng = random.Random(seed)
            vectors.append([rng.uniform(-1, 1) for _ in range(self.dims)])
        return vectors


class SiliconFlowEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
pytest tests/test_embeddings.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add agenticrag/embeddings.py tests/test_embeddings.py
git commit -m "feat: add embedding clients"
```

---

### Task 7: Chroma Retriever

**Files:**
- Create: `agenticrag/retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write failing retriever test**

Create `tests/test_retriever.py`:

```python
from pathlib import Path

from agenticrag.embeddings import FakeEmbeddingClient
from agenticrag.models import DocumentChunk
from agenticrag.retriever import ChromaRetriever


def test_retriever_adds_and_queries_chunks(tmp_path):
    retriever = ChromaRetriever(
        persist_dir=tmp_path / "chroma",
        embedding_client=FakeEmbeddingClient(dims=8),
        collection_name="test",
    )
    chunks = [
        DocumentChunk("doc1", Path("docs/a.md"), "A", "md", 0, 0, 0, "alpha content"),
        DocumentChunk("doc2", Path("docs/b.md"), "B", "md", 0, 0, 0, "beta content"),
    ]

    retriever.reset()
    retriever.add_chunks(chunks)
    results = retriever.query("alpha", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.content in {"alpha content", "beta content"}
    assert results[0].score >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests/test_retriever.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement retriever**

Create `agenticrag/retriever.py`:

```python
from __future__ import annotations

from pathlib import Path

import chromadb

from agenticrag.embeddings import EmbeddingClient
from agenticrag.models import DocumentChunk, RetrievedChunk


class ChromaRetriever:
    def __init__(
        self,
        persist_dir: Path,
        embedding_client: EmbeddingClient,
        collection_name: str = "agenticrag",
    ) -> None:
        self.persist_dir = persist_dir
        self.embedding_client = embedding_client
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(collection_name)

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        documents = [chunk.content for chunk in chunks]
        embeddings = self.embedding_client.embed_texts(documents)
        ids = [f"{chunk.doc_id}:{chunk.chunk_index}" for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        query_embedding = self.embedding_client.embed_texts([query])[0]
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        results: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            chunk = DocumentChunk(
                doc_id=str(metadata["doc_id"]),
                path=Path(str(metadata["path"])),
                title=str(metadata["title"]),
                filetype=str(metadata["filetype"]),
                chunk_index=int(metadata["chunk_index"]),
                line_start=int(metadata["line_start"]),
                line_end=int(metadata["line_end"]),
                content=str(document),
            )
            results.append(RetrievedChunk(chunk=chunk, score=float(distance)))
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
pytest tests/test_retriever.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add agenticrag/retriever.py tests/test_retriever.py
git commit -m "feat: add Chroma retriever"
```

---

### Task 8: Conversation State And Reference IDs

**Files:**
- Create: `agenticrag/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write failing state tests**

Create `tests/test_state.py`:

```python
from pathlib import Path

from agenticrag.models import DocumentChunk
from agenticrag.state import ConversationState


def test_assign_reference_ids_are_turn_scoped():
    state = ConversationState(user_query="问题")
    chunks = [
        DocumentChunk("doc1", Path("a.md"), "A", "md", 0, 0, 0, "alpha"),
        DocumentChunk("doc2", Path("b.md"), "B", "md", 0, 0, 0, "beta"),
    ]

    refs = state.assign_search_results(chunks)

    assert [ref.reference_id for ref in refs] == ["turn0search0", "turn0search1"]
    assert state.get_reference("turn0search1").chunk.content == "beta"


def test_summarize_compacts_unretained_tool_results():
    state = ConversationState(user_query="问题")
    chunk = DocumentChunk("doc1", Path("a.md"), "A", "md", 0, 0, 0, "alpha")
    state.assign_search_results([chunk])
    state.add_tool_result("open", "large payload", {"reference_ids": ["turn0search0"]})
    state.add_tool_result("open", "other payload", {"reference_ids": ["missing"]})

    state.summarize(["turn0search0"])

    assert state.tool_results[0].content == "large payload"
    assert "[compressed" in state.tool_results[1].content
    assert state.get_reference("turn0search0").chunk.content == "alpha"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_state.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement state**

Create `agenticrag/state.py`:

```python
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

    def assign_search_results(self, chunks: list[DocumentChunk]) -> list[Reference]:
        refs: list[Reference] = []
        for index, chunk in enumerate(chunks):
            reference_id = f"turn{self.turn_index}search{index}"
            ref = Reference(reference_id=reference_id, chunk=chunk)
            self.references[reference_id] = ref
            refs.append(ref)
        return refs

    def get_reference(self, reference_id: str) -> Reference:
        if reference_id not in self.references:
            raise KeyError(f"Unknown reference_id: {reference_id}")
        return self.references[reference_id]

    def add_tool_result(
        self,
        name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.tool_results.append(ToolResult(name=name, content=content, metadata=metadata or {}))
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
        for result in self.tool_results:
            refs = set(result.metadata.get("reference_ids", []))
            if refs and refs.isdisjoint(retained):
                result.content = f"[compressed {result.name} result unrelated to retained references]"
        self.messages = [
            message
            for message in self.messages
            if message.get("role") != "tool"
        ]
        for result in self.tool_results:
            self.messages.append({"role": "tool", "name": result.name, "content": result.content})
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
pytest tests/test_state.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add agenticrag/state.py tests/test_state.py
git commit -m "feat: manage conversation state"
```

---

### Task 9: Retrieval Tool Implementations

**Files:**
- Create: `agenticrag/tools/__init__.py`
- Create: `agenticrag/tools/retrieval.py`
- Create: `agenticrag/tools/schemas.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write failing tool tests**

Create `tests/test_tools.py`:

```python
import json
from pathlib import Path

from agenticrag.models import DocumentChunk, RetrievedChunk
from agenticrag.state import ConversationState
from agenticrag.tools.retrieval import RetrievalTools
from agenticrag.tools.schemas import TOOL_SCHEMAS


class StubRetriever:
    def query(self, query, top_k=10):
        chunk = DocumentChunk("doc1", Path("docs/a.md"), "A", "md", 0, 0, 2, "Alpha keyword content")
        return [RetrievedChunk(chunk=chunk, score=0.1)]


def write_cache(cache_dir):
    cache_dir.mkdir()
    payload = {
        "doc_id": "doc1",
        "path": "docs/a.md",
        "title": "A",
        "filetype": "md",
        "lines": ["First line", "Alpha keyword here", "Last line"],
    }
    (cache_dir / "doc1.json").write_text(json.dumps(payload), encoding="utf-8")


def test_search_assigns_reference_ids(tmp_path):
    write_cache(tmp_path)
    state = ConversationState(user_query="alpha")
    tools = RetrievalTools(StubRetriever(), tmp_path, state)

    content = tools.search(["alpha"])

    assert "turn0search0" in content
    assert "Alpha keyword content" in content
    assert state.get_reference("turn0search0").chunk.doc_id == "doc1"


def test_find_and_open_use_source_cache(tmp_path):
    write_cache(tmp_path)
    state = ConversationState(user_query="alpha")
    tools = RetrievalTools(StubRetriever(), tmp_path, state)
    tools.search(["alpha"])

    found = tools.find("turn0search0", ["keyword"])
    opened = tools.open("turn0search0", line_number=1)

    assert "Alpha keyword here" in found
    assert "Viewing lines [1-2] of 3 lines" in opened
    assert "1: Alpha keyword here" in opened


def test_tool_schemas_include_four_tools():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}

    assert names == {"search", "find", "open", "summarize"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_tools.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement schemas**

Create `agenticrag/tools/schemas.py`:

```python
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search across the enterprise corpus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 5,
                    }
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Find patterns within a referenced document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reference_id": {"type": "string"},
                    "patterns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["reference_id", "patterns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open",
            "description": "Open a line-numbered window in a referenced document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reference_id": {"type": "string"},
                    "line_number": {"type": "integer", "default": 0},
                },
                "required": ["reference_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Compress tool history while retaining selected references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_reference_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["candidate_reference_ids"],
            },
        },
    },
]
```

- [ ] **Step 4: Implement retrieval tools**

Create `agenticrag/tools/retrieval.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from agenticrag.models import DocumentChunk
from agenticrag.state import ConversationState


class RetrievalTools:
    def __init__(self, retriever, source_cache_dir: Path, state: ConversationState) -> None:
        self.retriever = retriever
        self.source_cache_dir = source_cache_dir
        self.state = state

    def _load_source(self, doc_id: str) -> dict:
        path = self.source_cache_dir / f"{doc_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing source cache for doc_id={doc_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def search(self, queries: list[str]) -> str:
        seen: set[tuple[str, str]] = set()
        chunks: list[DocumentChunk] = []
        for query in queries[:5]:
            for result in self.retriever.query(query, top_k=6):
                key = (result.chunk.doc_id, result.chunk.content)
                if key in seen:
                    continue
                seen.add(key)
                chunks.append(result.chunk)
                if len(chunks) >= 10:
                    break
            if len(chunks) >= 10:
                break

        refs = self.state.assign_search_results(chunks)
        lines = []
        for ref in refs:
            chunk = ref.chunk
            lines.append(
                "\n".join(
                    [
                        f"Reference ID: {ref.reference_id}",
                        f"Title: {chunk.title}",
                        f"Filename: {chunk.path.name}",
                        f"Filetype: {chunk.filetype}",
                        f"Lines: {chunk.line_start}-{chunk.line_end}",
                        f"Snippet: {chunk.content[:200]}",
                    ]
                )
            )
        content = "\n\n".join(lines) if lines else "No search results."
        self.state.add_tool_result("search", content, {"reference_ids": [ref.reference_id for ref in refs]})
        return content

    def find(self, reference_id: str, patterns: list[str]) -> str:
        ref = self.state.get_reference(reference_id)
        source = self._load_source(ref.chunk.doc_id)
        text = "\n".join(source["lines"])
        lower_text = text.lower()
        snippets: list[str] = []
        for pattern in patterns:
            lower_pattern = pattern.lower()
            start = 0
            matches = 0
            while matches < 2:
                index = lower_text.find(lower_pattern, start)
                if index < 0:
                    break
                begin = max(0, index - 50)
                end = min(len(text), index + len(pattern) + 50)
                snippet = text[begin:end].replace("\n", " ")
                if snippet not in snippets:
                    snippets.append(snippet)
                start = index + len(pattern)
                matches += 1
        content = "\n".join(snippets)[:66000] if snippets else "No matches found."
        self.state.add_tool_result("find", content, {"reference_ids": [reference_id]})
        return content

    def open(self, reference_id: str, line_number: int = 0) -> str:
        ref = self.state.get_reference(reference_id)
        source = self._load_source(ref.chunk.doc_id)
        lines = source["lines"]
        total = len(lines)
        start = min(max(line_number, 0), max(total - 1, 0))
        end = min(start + 1799, max(total - 1, 0))
        numbered = [f"{idx}: {lines[idx]}" for idx in range(start, end + 1)]
        content = f"Viewing lines [{start}-{end}] of {total} lines\n" + "\n".join(numbered)
        self.state.add_tool_result("open", content, {"reference_ids": [reference_id]})
        return content

    def summarize(self, candidate_reference_ids: list[str]) -> str:
        self.state.summarize(candidate_reference_ids)
        content = "Compressed tool history while retaining: " + ", ".join(candidate_reference_ids)
        self.state.add_tool_result("summarize", content, {"reference_ids": candidate_reference_ids})
        return content
```

Create `agenticrag/tools/__init__.py`:

```python
from agenticrag.tools.retrieval import RetrievalTools
from agenticrag.tools.schemas import TOOL_SCHEMAS

__all__ = ["RetrievalTools", "TOOL_SCHEMAS"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```powershell
pytest tests/test_tools.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add agenticrag/tools tests/test_tools.py
git commit -m "feat: implement retrieval tools"
```

---

### Task 10: Prompt Templates And Switcher

**Files:**
- Create: `agenticrag/prompts.py`
- Create: `agenticrag/switcher.py`
- Test: `tests/test_switcher.py`

- [ ] **Step 1: Write failing switcher tests**

Create `tests/test_switcher.py`:

```python
from agenticrag.switcher import parse_switcher_decision


def test_parse_switcher_decision_accepts_simple_json():
    assert parse_switcher_decision('{"route": "simple"}') == "simple"


def test_parse_switcher_decision_accepts_complex_json_with_text():
    assert parse_switcher_decision('Result:\n{"route": "complex", "reason": "multi-doc"}') == "complex"


def test_parse_switcher_decision_defaults_to_complex_for_unknown():
    assert parse_switcher_decision("not json") == "complex"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_switcher.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement prompts**

Create `agenticrag/prompts.py`:

```python
SWITCHER_PROMPT = """Classify the user query for a RAG system.
Return only JSON: {"route": "simple"} or {"route": "complex"}.
Use simple for single-intent factual questions.
Use complex for multi-step, multi-document, comparison, long-document, or ambiguous questions.
"""

SYSTEM_PROMPT = """# Overall Instructions
- Search before answering when uncertain.
- Progressively explore using 'find' or 'open' when snippets are insufficient.
- Reuse previous results rather than performing search again.
- Cite every time when information is used from tool outputs.

# When to Use 'search'
- Use 'search' as the primary tool across the enterprise corpus.
- It should be your first choice for any work-related query.
- Use it when users reference current/changing information, enterprise-specific terms, or acronyms.
- Use it to verify details rather than making assumptions.

# When to Use 'find'
- Use 'find' for in-document pattern search for relevant files from search results.
- Use it when search results snippets do not give enough details.
- Use it to get a focused view of a result in relation to certain terms.

# When to Use 'open'
- Use 'open' for windowed full content retrieval for relevant files from search results.
- Use it when search results snippets are insufficient.
- Use it to pull in more content from the most promising results.
- You can open multiple search results.
- Use the option to choose a line number close to the relevant content based on line-numbered document previews.

请默认使用中文回答，并使用可用的 Reference ID 标注证据。
"""

SIMPLE_RAG_PROMPT = """请基于给定检索片段回答用户问题。
必须引用片段中的 Reference ID。
如果证据不足，请明确说明。
"""

FORCE_FINAL_ANSWER_PROMPT = """FORCEFINALANSWER:
You have reached the maximum number of tool calls. Produce the best final answer using only the collected context.
Include citations with Reference IDs whenever possible.
"""
```

- [ ] **Step 4: Implement switcher parser**

Create `agenticrag/switcher.py`:

```python
from __future__ import annotations

import json
import re
from typing import Literal

from agenticrag.prompts import SWITCHER_PROMPT

Route = Literal["simple", "complex"]


def parse_switcher_decision(text: str) -> Route:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return "complex"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "complex"
    route = payload.get("route")
    return "simple" if route == "simple" else "complex"


def classify_query(llm_client, query: str) -> Route:
    response = llm_client.complete(
        messages=[
            {"role": "system", "content": SWITCHER_PROMPT},
            {"role": "user", "content": query},
        ]
    )
    return parse_switcher_decision(response)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```powershell
pytest tests/test_switcher.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add agenticrag/prompts.py agenticrag/switcher.py tests/test_switcher.py
git commit -m "feat: add prompts and query switcher"
```

---

### Task 11: DeepSeek LLM Adapter

**Files:**
- Create: `agenticrag/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write failing stream parsing test**

Create `tests/test_llm.py`:

```python
from agenticrag.llm import collect_stream_text


class Delta:
    def __init__(self, content):
        self.content = content


class Choice:
    def __init__(self, content):
        self.delta = Delta(content)


class Event:
    def __init__(self, content):
        self.choices = [Choice(content)]


def test_collect_stream_text_yields_content():
    events = [Event("你"), Event("好"), Event(None)]

    assert list(collect_stream_text(events)) == ["你", "好"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests/test_llm.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement LLM adapter**

Create `agenticrag/llm.py`:

```python
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from openai import OpenAI


def collect_stream_text(events: Iterable[Any]) -> Iterator[str]:
    for event in events:
        if not event.choices:
            continue
        content = getattr(event.choices[0].delta, "content", None)
        if content:
            yield content


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def complete(self, messages: list[dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content or ""

    def stream(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        events = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        yield from collect_stream_text(events)

    def tool_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=False,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
pytest tests/test_llm.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add agenticrag/llm.py tests/test_llm.py
git commit -m "feat: add DeepSeek LLM adapter"
```

---

### Task 12: Agentic Loop Control

**Files:**
- Create: `agenticrag/loop.py`
- Test: `tests/test_loop.py`

- [ ] **Step 1: Write failing loop tests**

Create `tests/test_loop.py`:

```python
from agenticrag.loop import should_force_completion


def test_should_force_completion_at_max_calls():
    assert should_force_completion(call_index=15, max_calls=15) is True
    assert should_force_completion(call_index=14, max_calls=15) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests/test_loop.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement loop helpers and orchestration shell**

Create `agenticrag/loop.py`:

```python
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

from agenticrag.prompts import FORCE_FINAL_ANSWER_PROMPT, SIMPLE_RAG_PROMPT, SYSTEM_PROMPT
from agenticrag.state import ConversationState
from agenticrag.tools.schemas import TOOL_SCHEMAS


def should_force_completion(call_index: int, max_calls: int) -> bool:
    return call_index >= max_calls


def stream_simple_rag(llm_client, query: str, search_context: str) -> Iterator[str]:
    messages = [
        {"role": "system", "content": SIMPLE_RAG_PROMPT},
        {"role": "user", "content": f"问题：{query}\n\n检索结果：\n{search_context}"},
    ]
    yield from llm_client.stream(messages)


def _message_from_response(response) -> Any:
    return response.choices[0].message


def run_agentic_loop(
    llm_client,
    state: ConversationState,
    tool_executor: Callable[[str, dict[str, Any]], str],
    max_calls: int,
    token_threshold: int,
    token_warning_ratio: float,
    status_writer: Callable[[str], None],
) -> Iterator[str]:
    state.add_message("system", SYSTEM_PROMPT)
    for call_index in range(1, max_calls + 1):
        state.maybe_add_token_warning(token_threshold, token_warning_ratio)
        if state.total_tokens() >= token_threshold:
            tool_executor("summarize", {"candidate_reference_ids": list(state.references.keys())})

        response = llm_client.tool_call(state.messages, TOOL_SCHEMAS)
        message = _message_from_response(response)
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments or "{}")
                status_writer(f"[tool] {name}")
                result = tool_executor(name, arguments)
                state.add_message("tool", result, tool_call_id=tool_call.id, name=name)
            continue

        content = message.content or ""
        if content:
            yield content
            return

    state.add_message("system", FORCE_FINAL_ANSWER_PROMPT)
    yield from llm_client.stream(state.messages)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
pytest tests/test_loop.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add agenticrag/loop.py tests/test_loop.py
git commit -m "feat: add agentic loop control"
```

---

### Task 13: CLI Index Command

**Files:**
- Create: `main.py`
- Modify: `agenticrag/ingest.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI argument test**

Create `tests/test_cli.py`:

```python
from main import build_parser


def test_parser_supports_index_and_ask():
    parser = build_parser()

    index_args = parser.parse_args(["index"])
    ask_args = parser.parse_args(["ask", "什么是AgenticRAG？"])

    assert index_args.command == "index"
    assert ask_args.command == "ask"
    assert ask_args.query == "什么是AgenticRAG？"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests/test_cli.py -v
```

Expected: FAIL with missing `main.py` or `build_parser`.

- [ ] **Step 3: Add ingest corpus helper**

Append to `agenticrag/ingest.py`:

```python
def parse_corpus(docs_dir: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in scan_documents(docs_dir):
        try:
            chunks.extend(parse_document(path))
        except Exception as exc:
            print(f"[warn] failed to parse {path}: {exc}")
    return chunks
```

- [ ] **Step 4: Implement CLI parser and index command**

Create `main.py`:

```python
from __future__ import annotations

import argparse
import sys

from agenticrag.config import ConfigError, load_config
from agenticrag.embeddings import SiliconFlowEmbeddingClient
from agenticrag.ingest import parse_corpus, write_source_cache
from agenticrag.retriever import ChromaRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgenticRAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("index", help="Index local documents")
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query")
    return parser


def run_index() -> int:
    config = load_config()
    embedding_client = SiliconFlowEmbeddingClient(
        api_key=config.siliconflow_api_key,
        base_url=config.siliconflow_base_url,
        model=config.siliconflow_embedding_model,
    )
    retriever = ChromaRetriever(config.chroma_dir, embedding_client)
    chunks = parse_corpus(config.docs_dir)
    if not chunks:
        print(f"No supported documents found under {config.docs_dir}")
        return 1
    retriever.reset()
    retriever.add_chunks(chunks)
    write_source_cache(config.source_cache_dir, chunks)
    print(f"Indexed {len(chunks)} chunks from {config.docs_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            return run_index()
        if args.command == "ask":
            from agenticrag.loop import run_ask

            return run_ask(args.query)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run parser test**

Run:

```powershell
pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add main.py agenticrag/ingest.py tests/test_cli.py
git commit -m "feat: add CLI index command"
```

---

### Task 14: CLI Ask Command

**Files:**
- Modify: `agenticrag/loop.py`
- Modify: `main.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add ask dispatch test**

Append to `tests/test_cli.py`:

```python
def test_main_dispatches_ask(monkeypatch):
    calls = []

    def fake_run_ask(query):
        calls.append(query)
        return 0

    import agenticrag.loop
    from main import main

    monkeypatch.setattr(agenticrag.loop, "run_ask", fake_run_ask)

    assert main(["ask", "问题"]) == 0
    assert calls == ["问题"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests/test_cli.py::test_main_dispatches_ask -v
```

Expected: FAIL because `run_ask` is not defined.

- [ ] **Step 3: Implement ask orchestration**

Append to `agenticrag/loop.py`:

```python
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
    tools = RetrievalTools(retriever, config.source_cache_dir, state)

    route = classify_query(llm_client, query)
    if route == "simple":
        context = tools.search([query])
        for chunk in stream_simple_rag(llm_client, query, context):
            print(chunk, end="", flush=True)
        print()
        return 0

    def execute_tool(name: str, arguments: dict[str, Any]) -> str:
        if name == "search":
            return tools.search(arguments.get("queries", []))
        if name == "find":
            return tools.find(arguments["reference_id"], arguments.get("patterns", []))
        if name == "open":
            return tools.open(arguments["reference_id"], int(arguments.get("line_number", 0)))
        if name == "summarize":
            return tools.summarize(arguments.get("candidate_reference_ids", []))
        return f"Unknown tool: {name}"

    for chunk in run_agentic_loop(
        llm_client=llm_client,
        state=state,
        tool_executor=execute_tool,
        max_calls=config.max_calls,
        token_threshold=config.token_threshold,
        token_warning_ratio=config.token_warning_ratio,
        status_writer=lambda text: print(text, flush=True),
    ):
        print(chunk, end="", flush=True)
    print()
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add agenticrag/loop.py main.py tests/test_cli.py
git commit -m "feat: add streaming ask command"
```

---

### Task 15: Focused Full Test Run

**Files:**
- Modify only files needed to fix test failures discovered in this task.

- [ ] **Step 1: Run full pytest suite**

Run:

```powershell
pytest -v
```

Expected: all tests PASS. If a test fails because of a mismatch introduced by an earlier task, fix the smallest affected function and rerun the failing test first.

- [ ] **Step 2: Run import and CLI help checks**

Run:

```powershell
python main.py --help
python main.py ask --help
```

Expected: both commands print argparse help text and exit with code 0.

- [ ] **Step 3: Commit fixes if any**

If files changed:

```powershell
git add agenticrag tests main.py
git commit -m "test: stabilize AgenticRAG MVP tests"
```

If no files changed, do not create an empty commit.

---

### Task 16: Manual Index And Streaming Smoke Test

**Files:**
- Modify: `README.md`
- Modify only implementation files needed to fix smoke-test failures.

- [ ] **Step 1: Create README usage draft**

Create `README.md`:

```markdown
# Microsoft AgenticRAG Reproduction

Python MVP reproduction of AgenticRAG for local enterprise-style documents.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

Fill `DEEPSEEK_API_KEY` and `SILICONFLOW_API_KEY` in `.env`.

## Index

```powershell
python main.py index
```

## Ask

```powershell
python main.py ask "AgenticRAG 的核心模块有哪些？"
```

Final answers stream to the terminal. Intermediate AgenticRAG tool calls print compact status lines.
```
```

- [ ] **Step 2: Run manual index with real keys**

Run:

```powershell
python main.py index
```

Expected: prints `Indexed N chunks from docs` with `N > 0`.

- [ ] **Step 3: Run simple ask smoke test**

Run:

```powershell
python main.py ask "AgenticRAG 的核心模块有哪些？"
```

Expected: answer streams progressively and includes citations or Reference IDs.

- [ ] **Step 4: Run complex ask smoke test**

Run:

```powershell
python main.py ask "请对比 AgenticRAG 的 search、find、open、summarize 四个工具的作用，并说明它们如何协同。"
```

Expected: output includes compact `[tool]` status lines before the final streamed answer, or the switcher routes simple and still produces a coherent cited answer.

- [ ] **Step 5: Commit README and smoke-test fixes**

```powershell
git add README.md agenticrag main.py tests
git commit -m "docs: add AgenticRAG MVP usage"
```

---

### Task 17: Final Push

**Files:**
- No code changes expected unless status reveals missed tracked files.

- [ ] **Step 1: Check working tree**

Run:

```powershell
git status --short
```

Expected: only intentionally untracked research/source materials remain, such as local PDF/MHTML files. There should be no modified tracked implementation files.

- [ ] **Step 2: Push commits**

Run:

```powershell
git push
```

Expected: local `main` pushes cleanly to `origin/main`.

- [ ] **Step 3: Report verification**

Report:

- Latest commit hash.
- Tests run.
- Whether manual indexing and ask smoke tests passed.
- Any untracked research files intentionally left out of Git.

---

## Self-Review

Spec coverage:

- CLI `index` and `ask`: Tasks 13, 14, 16.
- Local Markdown/PDF ingestion: Tasks 4, 5, 13.
- Chroma retrieval: Task 7.
- SiliconFlow embeddings: Task 6.
- DeepSeek and streaming output: Tasks 11, 14, 16.
- Switcher: Task 10.
- Agentic Loop with max calls and forced completion: Task 12.
- Retrieval tools `search`, `find`, `open`, `summarize`: Task 9.
- Conversation state, token estimates, Reference IDs, compression: Task 8.
- Error handling for configuration and parsing: Tasks 2, 5, 13.
- Tests and smoke checks: Tasks 1-17.

Placeholder scan:

- This plan intentionally contains no placeholder markers or unspecified implementation steps.

Type consistency:

- `DocumentChunk`, `RetrievedChunk`, `Reference`, `ToolResult`, `ConversationState`, `RetrievalTools`, `DeepSeekClient`, and `ChromaRetriever` are introduced before downstream tasks depend on them.
