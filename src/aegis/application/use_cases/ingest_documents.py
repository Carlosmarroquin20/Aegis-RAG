"""
IngestDocumentsUseCase — orchestrates the full ingestion pipeline.

Pipeline stages:
  1. MIME type detection from magic bytes (not Content-Type header)
  2. File size enforcement
  3. Format-specific parsing → list[RawDocument]
  4. Sentence-aware chunking → list[Document]
  5. Content deduplication via hash comparison (skip existing IDs)
  6. Vector store upsert

Security considerations:
  - MIME type is verified from magic bytes to prevent file extension spoofing.
  - File size is bounded before parsing to prevent decompression bombs (PDF/DOCX).
  - Content is not evaluated by the SecurityGateway — ingested documents are
    trusted source material, not user queries. The gateway protects the query
    path; indirect injection via document content is mitigated at the LLM layer
    by the hardened system prompt in OllamaAdapter.
"""
from __future__ import annotations

import structlog

from aegis.application.dtos.ingestion_dtos import IngestRequest, IngestResponse, UploadedFile
from aegis.domain.models.ingestion import IngestionResult
from aegis.domain.ports.document_parser import ParseError
from aegis.domain.ports.vector_store import VectorStorePort
from aegis.domain.services.chunking_service import ChunkingService

logger = structlog.get_logger(__name__)

# 50 MB hard ceiling. Larger files should be pre-split outside the API.
_MAX_FILE_BYTES: int = 50 * 1024 * 1024


class FileTooLargeError(Exception):
    pass


class UnsupportedFileTypeError(Exception):
    pass


class IngestDocumentsUseCase:
    def __init__(
        self,
        vector_store: VectorStorePort,
        parser_registry: object,  # ParserRegistry — typed as object to avoid circular import
        default_collection: str,
    ) -> None:
        self._vector_store = vector_store
        self._registry = parser_registry
        self._default_collection = default_collection

    async def execute(
        self,
        uploaded_file: UploadedFile,
        request: IngestRequest,
    ) -> IngestResponse:
        log = logger.bind(filename=uploaded_file.filename)

        # ── Stage 1: File size guard ───────────────────────────────────────────
        if len(uploaded_file.content) > _MAX_FILE_BYTES:
            raise FileTooLargeError(
                f"File '{uploaded_file.filename}' exceeds the "
                f"{_MAX_FILE_BYTES // (1024 * 1024)} MB limit."
            )

        # ── Stage 2: MIME type detection from magic bytes ──────────────────────
        # Importing here to keep the use case testable without the full infra stack.
        from aegis.infrastructure.parsers.parser_registry import ParserRegistry

        registry: ParserRegistry = self._registry  # type: ignore[assignment]
        mime_type = registry.detect_mime_type(uploaded_file.content, uploaded_file.filename)

        parser = registry.get_parser(mime_type)
        if parser is None:
            raise UnsupportedFileTypeError(
                f"No parser available for MIME type '{mime_type}' "
                f"(file: '{uploaded_file.filename}')."
            )

        log.info("ingestion.started", mime_type=mime_type)

        # ── Stage 3: Parsing ───────────────────────────────────────────────────
        warnings: list[str] = []
        try:
            raw_documents = parser.parse(uploaded_file.content, uploaded_file.filename)
        except ParseError as exc:
            log.error("ingestion.parse_failed", reason=exc.reason)
            raise

        log.debug("ingestion.parsed", raw_doc_count=len(raw_documents))

        # ── Stage 4: Chunking ──────────────────────────────────────────────────
        chunker = ChunkingService(
            chunk_size=request.chunk_size,
            overlap=request.overlap,
        )
        all_chunks = []
        for raw_doc in raw_documents:
            try:
                all_chunks.extend(chunker.chunk(raw_doc))
            except Exception as exc:  # noqa: BLE001
                msg = f"Chunking error on page {raw_doc.page_number}: {exc}"
                warnings.append(msg)
                log.warning("ingestion.chunk_error", error=str(exc))

        log.debug("ingestion.chunked", chunk_count=len(all_chunks))

        if not all_chunks:
            warnings.append("No chunks produced — file may be empty or unreadable.")
            return IngestResponse(
                source=uploaded_file.filename,
                chunks_created=0,
                duplicates_skipped=0,
                warnings=warnings,
                collection=request.collection or self._default_collection,
            )

        # ── Stage 5: Deduplication ─────────────────────────────────────────────
        # Chunks use content-addressed IDs: re-uploading the same file produces
        # the same IDs. The vector store's upsert is idempotent, but we track
        # how many were new vs. already present for observability.
        existing_ids: set[str] = set()
        try:
            # Peek at existing IDs by doing a dummy similarity search; production
            # deployments should use a dedicated ID-existence check if the store
            # supports it (e.g., ChromaDB's .get(ids=[...])).
            from aegis.infrastructure.vector_stores.chromadb_adapter import ChromaDBAdapter

            if isinstance(self._vector_store, ChromaDBAdapter):
                candidate_ids = [c.id for c in all_chunks]
                existing_ids = await self._get_existing_ids(candidate_ids)
        except Exception as exc:  # noqa: BLE001
            # Deduplication is best-effort; a failure here does not block the upsert
            # because add_documents() is idempotent on duplicate IDs.
            log.warning("ingestion.dedup_failed", error=str(exc))

        new_chunks = [c for c in all_chunks if c.id not in existing_ids]
        duplicates = len(all_chunks) - len(new_chunks)

        # ── Stage 6: Upsert ────────────────────────────────────────────────────
        if new_chunks:
            await self._vector_store.add_documents(new_chunks)

        result = IngestionResult(
            source=uploaded_file.filename,
            chunks_created=len(new_chunks),
            duplicates_skipped=duplicates,
            warnings=warnings,
        )
        log.info(
            "ingestion.complete",
            chunks_created=result.chunks_created,
            duplicates_skipped=result.duplicates_skipped,
        )

        return IngestResponse(
            source=result.source,
            chunks_created=result.chunks_created,
            duplicates_skipped=result.duplicates_skipped,
            warnings=result.warnings,
            collection=request.collection or self._default_collection,
        )

    async def _get_existing_ids(self, ids: list[str]) -> set[str]:
        """
        Checks the vector store for pre-existing chunk IDs.
        Returns only those IDs that are already indexed.
        This is a best-effort deduplication — not a correctness guarantee.
        """
        from aegis.infrastructure.vector_stores.chromadb_adapter import ChromaDBAdapter

        store = self._vector_store
        if not isinstance(store, ChromaDBAdapter):
            return set()

        # ChromaDB collection.get() returns only IDs that exist.
        assert store._collection is not None  # noqa: SLF001
        result = store._collection.get(ids=ids, include=[])  # noqa: SLF001
        return set(result.get("ids", []))
