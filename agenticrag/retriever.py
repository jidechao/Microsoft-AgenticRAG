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
