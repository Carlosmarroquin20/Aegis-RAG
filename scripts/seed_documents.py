"""
Seed script: loads documents from a local directory into the vector store.

Usage:
    uv run python scripts/seed_documents.py --source-dir ./data/docs

The script reads .txt and .md files, chunks them, and upserts them into
the configured ChromaDB collection.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import textwrap
from pathlib import Path

from aegis.config import get_settings
from aegis.domain.models.document import Document
from aegis.interface.api.dependencies import get_chromadb_adapter


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Splits text into overlapping chunks by character count.
    Overlap preserves context at chunk boundaries for better retrieval.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


async def seed(source_dir: Path) -> None:
    cfg = get_settings()
    adapter = get_chromadb_adapter(cfg)
    await adapter.initialize()

    documents: list[Document] = []
    extensions = {".txt", ".md"}

    for file_path in source_dir.rglob("*"):
        if file_path.suffix not in extensions:
            continue

        content = file_path.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk_text(content)

        for i, chunk in enumerate(chunks):
            doc_id = hashlib.sha256(f"{file_path}:{i}".encode()).hexdigest()[:32]
            documents.append(
                Document(
                    id=doc_id,
                    content=chunk,
                    metadata={
                        "source": str(file_path.relative_to(source_dir)),
                        "chunk_index": str(i),
                    },
                )
            )

    if not documents:
        print(f"No .txt or .md files found in {source_dir}")
        return

    await adapter.add_documents(documents)
    print(f"Seeded {len(documents)} chunks from {source_dir} into '{cfg.chroma_collection}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed documents into Aegis-RAG vector store.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/docs"),
        help="Directory containing .txt and .md files to index.",
    )
    args = parser.parse_args()
    asyncio.run(seed(args.source_dir))


if __name__ == "__main__":
    main()
