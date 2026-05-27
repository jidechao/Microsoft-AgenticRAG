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


def test_build_parser_parses_chat_command():
    parser = cli.build_parser()

    args = parser.parse_args(["chat"])

    assert args.command == "chat"


def test_main_dispatches_ask(monkeypatch):
    query = "\u4ec0\u4e48\u662fAgenticRAG\uff1f"
    calls = []

    def fake_run_ask(value):
        calls.append(value)
        return 0

    monkeypatch.setattr("agenticrag.loop.run_ask", fake_run_ask, raising=False)

    assert cli.main(["ask", query]) == 0
    assert calls == [query]


def test_main_dispatches_chat(monkeypatch):
    calls = []

    def fake_run_chat():
        calls.append("chat")
        return 0

    monkeypatch.setattr("agenticrag.chat.run_chat", fake_run_chat)

    assert cli.main(["chat"]) == 0
    assert calls == ["chat"]


def test_main_configures_stdio_before_dispatch(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "configure_stdio", lambda: calls.append("configured"))
    monkeypatch.setattr(cli, "run_index", lambda: calls.append("index") or 0)

    assert cli.main(["index"]) == 0
    assert calls == ["configured", "index"]


def test_run_ask_simple_route_streams_chunks(monkeypatch, capsys, tmp_path):
    from agenticrag import loop

    query = "\u4ec0\u4e48\u662fAgenticRAG\uff1f"
    config = SimpleNamespace(
        deepseek_api_key="deepseek-key",
        deepseek_base_url="https://deepseek.test",
        deepseek_model="deepseek-model",
        siliconflow_api_key="silicon-key",
        siliconflow_base_url="https://silicon.test",
        siliconflow_embedding_model="embedding-model",
        chroma_dir=tmp_path / "chroma",
        source_cache_dir=tmp_path / "cache",
        max_calls=3,
        token_threshold=10_000,
        token_warning_ratio=0.8,
    )
    calls = []
    stream_inputs = []

    class FakeDeepSeekClient:
        def __init__(self, **kwargs):
            calls.append(("llm", kwargs))

    class FakeEmbeddingClient:
        def __init__(self, **kwargs):
            calls.append(("embedding", kwargs))

    class FakeRetriever:
        def __init__(self, persist_dir, embedding_client):
            calls.append(("retriever", persist_dir, embedding_client))

    class FakeTools:
        def __init__(self, *, retriever, state, source_cache_dir):
            self.retriever = retriever
            self.state = state
            self.source_cache_dir = source_cache_dir
            calls.append(("tools", retriever, state.user_query, source_cache_dir))

        def search(self, queries):
            calls.append(("search", queries))
            return "context"

    def fake_stream_simple_rag(llm_client, value, context):
        stream_inputs.append((llm_client, value, context))
        yield "part 1"
        yield " and part 2"

    monkeypatch.setattr("agenticrag.config.load_config", lambda: config)
    monkeypatch.setattr("agenticrag.llm.DeepSeekClient", FakeDeepSeekClient)
    monkeypatch.setattr("agenticrag.embeddings.SiliconFlowEmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr("agenticrag.retriever.ChromaRetriever", FakeRetriever)
    monkeypatch.setattr("agenticrag.switcher.classify_query", lambda llm, value: "simple")
    monkeypatch.setattr("agenticrag.tools.retrieval.RetrievalTools", FakeTools)
    monkeypatch.setattr(loop, "stream_simple_rag", fake_stream_simple_rag)
    monkeypatch.setattr(
        loop,
        "run_agentic_loop",
        lambda *args, **kwargs: pytest.fail("agentic loop should not run for simple route"),
    )

    assert loop.run_ask(query) == 0

    captured = capsys.readouterr()
    assert captured.out == "part 1 and part 2\n"
    assert ("search", [query]) in calls
    assert stream_inputs[0][1:] == (query, "context")


def test_run_ask_complex_route_streams_status_and_chunks(monkeypatch, capsys, tmp_path):
    from agenticrag import loop

    query = "\u8bf4\u660eAgenticRAG\u5b9e\u73b0\u7ec6\u8282"
    config = SimpleNamespace(
        deepseek_api_key="deepseek-key",
        deepseek_base_url="https://deepseek.test",
        deepseek_model="deepseek-model",
        siliconflow_api_key="silicon-key",
        siliconflow_base_url="https://silicon.test",
        siliconflow_embedding_model="embedding-model",
        chroma_dir=tmp_path / "chroma",
        source_cache_dir=tmp_path / "cache",
        max_calls=2,
        token_threshold=500,
        token_warning_ratio=0.7,
    )
    calls = []

    class FakeDeepSeekClient:
        def __init__(self, **kwargs):
            calls.append(("llm", kwargs))

    class FakeEmbeddingClient:
        def __init__(self, **kwargs):
            calls.append(("embedding", kwargs))

    class FakeRetriever:
        def __init__(self, persist_dir, embedding_client):
            calls.append(("retriever", persist_dir, embedding_client))

    class FakeTools:
        def __init__(self, *, retriever, state, source_cache_dir):
            self.state = state
            calls.append(("tools", retriever, state.user_query, source_cache_dir))

        def search(self, queries):
            calls.append(("search", queries))
            return "search result"

        def find(self, reference_id, patterns):
            calls.append(("find", reference_id, patterns))
            return "find result"

        def open(self, reference_id, line_number=0):
            calls.append(("open", reference_id, line_number))
            return "open result"

        def summarize(self, candidate_reference_ids):
            calls.append(("summarize", candidate_reference_ids))
            return "summary result"

    def fake_run_agentic_loop(
        llm_client,
        state,
        tool_executor,
        max_calls,
        token_threshold,
        token_warning_ratio,
        status_writer,
    ):
        calls.append(
            (
                "agentic",
                state.user_query,
                max_calls,
                token_threshold,
                token_warning_ratio,
            )
        )
        status_writer("[tool] search")
        assert tool_executor("search", {"queries": [query]}) == "search result"
        assert tool_executor("find", {"reference_id": "ref-1", "patterns": ["agent"]}) == "find result"
        assert tool_executor("open", {"reference_id": "ref-1", "line_number": 9}) == "open result"
        assert tool_executor("summarize", {"candidate_reference_ids": ["ref-1"]}) == "summary result"
        assert "[tool error] search" in tool_executor("search", {"_error": "invalid tool arguments"})
        assert "[tool error] unknown" in tool_executor("unknown", {})
        yield "complex "
        yield "answer"

    monkeypatch.setattr("agenticrag.config.load_config", lambda: config)
    monkeypatch.setattr("agenticrag.llm.DeepSeekClient", FakeDeepSeekClient)
    monkeypatch.setattr("agenticrag.embeddings.SiliconFlowEmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr("agenticrag.retriever.ChromaRetriever", FakeRetriever)
    monkeypatch.setattr("agenticrag.switcher.classify_query", lambda llm, value: "complex")
    monkeypatch.setattr("agenticrag.tools.retrieval.RetrievalTools", FakeTools)
    monkeypatch.setattr(loop, "run_agentic_loop", fake_run_agentic_loop)

    assert loop.run_ask(query) == 0

    captured = capsys.readouterr()
    assert captured.out == "[tool] search\ncomplex answer\n"
    assert ("agentic", query, 2, 500, 0.7) in calls


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
