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
