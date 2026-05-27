from main import build_parser


def test_build_parser_parses_index_and_ask_commands():
    parser = build_parser()
    query = "\u4ec0\u4e48\u662fAgenticRAG\uff1f"

    index_args = parser.parse_args(["index"])
    ask_args = parser.parse_args(["ask", query])

    assert index_args.command == "index"
    assert ask_args.command == "ask"
    assert ask_args.query == query
