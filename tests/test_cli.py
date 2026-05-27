from pathlib import Path
from types import SimpleNamespace

import pytest

import main as cli
from agenticrag.config import ConfigError
from agenticrag.ingest import parse_corpus


def test_build_parser_parses_index_and_ask_commands():
    parser = cli.build_parser()
    query = "\u4ec0\u4e48\u662fAgenticRAG\uff1f"

    index_args = parser.parse_args(["index"])
    ask_args = parser.parse_args(["ask", query])

    assert index_args.command == "index"
    assert ask_args.command == "ask"
    assert ask_args.query == query


def test_main_returns_placeholder_for_unimplemented_ask(capsys):
    query = "\u4ec0\u4e48\u662fAgenticRAG\uff1f"

    exit_code = cli.main(["ask", query])

    assert exit_code == 1
    assert "ask command is not implemented yet" in capsys.readouterr().err


def test_main_dispatches_ask_when_run_ask_exists(monkeypatch):
    query = "\u4ec0\u4e48\u662fAgenticRAG\uff1f"
    calls = []

    def fake_run_ask(value):
        calls.append(value)
        return 0

    monkeypatch.setattr("agenticrag.loop.run_ask", fake_run_ask, raising=False)

    assert cli.main(["ask", query]) == 0
    assert calls == [query]


def test_main_reports_config_error(monkeypatch, capsys):
    def fail_index():
        raise ConfigError("missing key")

    monkeypatch.setattr(cli, "run_index", fail_index)

    assert cli.main(["index"]) == 2
    assert "Configuration error: missing key" in capsys.readouterr().err


def test_run_index_empty_corpus_does_not_create_clients(monkeypatch, capsys, tmp_path):
    config = SimpleNamespace(
        docs_dir=tmp_path,
        chroma_dir=tmp_path / "chroma",
        source_cache_dir=tmp_path / "cache",
        siliconflow_api_key="key",
        siliconflow_base_url="https://example.test",
        siliconflow_embedding_model="model",
    )

    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "has_supported_documents", lambda docs_dir: False)
    monkeypatch.setattr(cli, "parse_corpus", lambda docs_dir: [])
    monkeypatch.setattr(
        cli,
        "SiliconFlowEmbeddingClient",
        lambda **kwargs: pytest.fail("embedding client should not be created"),
    )
    monkeypatch.setattr(
        cli,
        "ChromaRetriever",
        lambda *args, **kwargs: pytest.fail("retriever should not be created"),
    )

    assert cli.run_index() == 1
    assert f"No supported documents found under {tmp_path}" in capsys.readouterr().out


def test_run_index_parse_failures_report_zero_chunks(monkeypatch, capsys, tmp_path):
    config = SimpleNamespace(
        docs_dir=tmp_path,
        chroma_dir=tmp_path / "chroma",
        source_cache_dir=tmp_path / "cache",
        siliconflow_api_key="key",
        siliconflow_base_url="https://example.test",
        siliconflow_embedding_model="model",
    )

    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "has_supported_documents", lambda docs_dir: True)
    monkeypatch.setattr(cli, "parse_corpus", lambda docs_dir: [])
    monkeypatch.setattr(
        cli,
        "SiliconFlowEmbeddingClient",
        lambda **kwargs: pytest.fail("embedding client should not be created"),
    )
    monkeypatch.setattr(
        cli,
        "ChromaRetriever",
        lambda *args, **kwargs: pytest.fail("retriever should not be created"),
    )

    assert cli.run_index() == 1
    assert f"No chunks parsed from supported documents under {tmp_path}" in capsys.readouterr().out


def test_parse_corpus_writes_parse_warnings_to_stderr(monkeypatch, capsys, tmp_path):
    path = tmp_path / "bad.md"

    monkeypatch.setattr("agenticrag.ingest.scan_documents", lambda docs_dir: [path])

    def fail_parse(document_path: Path):
        raise RuntimeError("boom")

    monkeypatch.setattr("agenticrag.ingest.parse_document", fail_parse)

    assert parse_corpus(tmp_path) == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"[warn] failed to parse {path}: boom" in captured.err
