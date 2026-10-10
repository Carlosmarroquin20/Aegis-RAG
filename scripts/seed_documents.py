"""
Seed script: indexes every supported document under a directory.

Usage:
    uv run python scripts/seed_documents.py --source-dir ./data/docs

Files go through ``IngestDocumentsUseCase`` — the same pipeline the HTTP API
uses — so a seeded corpus is indistinguishable from an uploaded one: identical
type detection, parsing, chunking and content-addressed chunk IDs. That last
part is what makes seeding idempotent and lets a document seeded here be
recognised as a duplicate if it is later uploaded through the API.

Requires the vector store to be reachable (``docker compose up -d chromadb``)
and uses the same environment configuration as the application.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

from aegis.application.dtos.ingestion_dtos import IngestRequest, UploadedFile
from aegis.application.use_cases.ingest_documents import (
    FileTooLargeError,
    IngestDocumentsUseCase,
    UnsupportedFileTypeError,
)
from aegis.domain.ports.document_parser import ParseError

# Extensions worth attempting. The use case still decides the real type from the
# file's content; this only avoids feeding it images, archives and the like.
SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".txt", ".md", ".markdown", ".pdf", ".docx"})


@dataclass
class SeedSummary:
    files_indexed: int = 0
    chunks_created: int = 0
    duplicates_skipped: int = 0
    # (relative path, reason) for every file that could not be ingested.
    failures: list[tuple[str, str]] = field(default_factory=list)


async def seed(
    source_dir: Path,
    use_case: IngestDocumentsUseCase,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
) -> SeedSummary:
    """
    Ingests every supported file under ``source_dir``, recursively.

    A file that fails to parse is recorded and skipped rather than aborting the
    run: one corrupt PDF should not prevent the rest of a corpus from loading.
    """
    request = IngestRequest(chunk_size=chunk_size, overlap=overlap)
    summary = SeedSummary()

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        # The relative path is the document's ``source``: stable across machines
        # and unambiguous when two directories hold a file with the same name.
        source = path.relative_to(source_dir).as_posix()
        try:
            result = await use_case.execute(
                UploadedFile(filename=source, content=path.read_bytes()), request
            )
        except (ParseError, UnsupportedFileTypeError, FileTooLargeError) as exc:
            summary.failures.append((source, str(exc)))
            continue

        summary.files_indexed += 1
        summary.chunks_created += result.chunks_created
        summary.duplicates_skipped += result.duplicates_skipped

    return summary


async def _run(source_dir: Path, chunk_size: int, overlap: int) -> int:
    # Imported here so the module (and ``seed``) can be used without the heavy
    # wiring, which loads the embedding model.
    from aegis.interface.api.dependencies import get_chromadb_adapter, get_ingest_use_case

    await get_chromadb_adapter().initialize()
    summary = await seed(source_dir, get_ingest_use_case(), chunk_size=chunk_size, overlap=overlap)

    print(
        f"Indexed {summary.files_indexed} file(s): {summary.chunks_created} new chunk(s), "
        f"{summary.duplicates_skipped} already present."
    )
    for source, reason in summary.failures:
        print(f"  skipped {source}: {reason}", file=sys.stderr)

    if summary.files_indexed == 0 and not summary.failures:
        print(f"No supported files found under {source_dir}.", file=sys.stderr)
        return 1
    return 1 if summary.failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed documents into the Aegis-RAG vector store.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/docs"),
        help="Directory to index recursively (TXT, MD, PDF, DOCX).",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        parser.error(f"--source-dir is not a directory: {args.source_dir}")

    sys.exit(asyncio.run(_run(args.source_dir, args.chunk_size, args.overlap)))


if __name__ == "__main__":
    main()
