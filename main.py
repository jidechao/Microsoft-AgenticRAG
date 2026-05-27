from __future__ import annotations

import argparse
import sys

from agenticrag.config import ConfigError
from agenticrag.config import load_config
from agenticrag.embeddings import SiliconFlowEmbeddingClient
from agenticrag.ingest import parse_corpus
from agenticrag.ingest import write_source_cache
from agenticrag.retriever import ChromaRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgenticRAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index")
    ask_parser = subparsers.add_parser("ask")
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
