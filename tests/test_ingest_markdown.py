from agenticrag.ingest import chunk_text, parse_markdown


def test_chunk_text_respects_size_and_overlap():
    chunks = chunk_text(
        "\u7b2c\u4e00\u53e5\u3002" * 200,
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
    combined_content = " ".join(chunk.content for chunk in chunks)
    assert "Intro" in combined_content
    assert "Body line" in combined_content
