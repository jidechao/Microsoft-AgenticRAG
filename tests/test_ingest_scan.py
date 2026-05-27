from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_scan_documents_includes_pdf_files(tmp_path):
    (tmp_path / "a.pdf").write_text("pdf", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("skip", encoding="utf-8")

    paths = scan_documents(tmp_path)

    assert paths == [tmp_path / "a.pdf", tmp_path / "b.md"]


def test_parse_document_dispatches_pdf(monkeypatch, tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4")
    fake_chunk = SimpleNamespace(marker="pdf")
    pdf_mock = MagicMock()
    pdf_mock.pages = [SimpleNamespace(extract_text=MagicMock(return_value="Hello PDF"))]
    pdf_context = MagicMock()
    pdf_context.__enter__.return_value = pdf_mock
    pdf_context.__exit__.return_value = False
    open_mock = MagicMock(return_value=pdf_context)

    monkeypatch.setattr("pdfplumber.open", open_mock)

    chunks = parse_document(path)

    assert open_mock.call_args.args == (path,)
    assert chunks
    assert chunks[0].filetype == "pdf"
    assert chunks[0].title == "a"
    assert chunks[0].content.startswith("[page 1]\nHello PDF")


def test_parse_pdf_ignores_blank_pages(monkeypatch, tmp_path):
    path = tmp_path / "blank.pdf"
    path.write_bytes(b"%PDF-1.4")
    pdf_mock = MagicMock()
    pdf_mock.pages = [
        SimpleNamespace(extract_text=MagicMock(return_value="   ")),
        SimpleNamespace(extract_text=MagicMock(return_value=None)),
    ]
    pdf_context = MagicMock()
    pdf_context.__enter__.return_value = pdf_mock
    pdf_context.__exit__.return_value = False
    open_mock = MagicMock(return_value=pdf_context)

    monkeypatch.setattr("pdfplumber.open", open_mock)

    chunks = parse_document(path)

    assert chunks == []
