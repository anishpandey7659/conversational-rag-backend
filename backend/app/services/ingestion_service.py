import logging
from uuid import UUID

from app.core.exceptions import IntegrationUnavailableError, NotFoundError
from app.core.settings import Settings
from app.integrations.qdrant import QdrantVectorStore
from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import ChunkingStrategy, DocumentStatus
from app.services.chunking_service import split_text
from app.services.embedding_service import EmbeddingService
from app.services.extraction_service import ExtractionService, validate_upload

logger = logging.getLogger(__name__)


class IngestionService:
    """Coordinate document validation, extraction, embedding, and storage."""

    def __init__(
        self,
        repository: DocumentRepository,
        extraction: ExtractionService,
        embeddings: EmbeddingService,
        vectors: QdrantVectorStore,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.extraction = extraction
        self.embeddings = embeddings
        self.vectors = vectors
        self.settings = settings

    async def ingest(
        self,
        filename: str | None,
        content_type: str | None,
        content: bytes,
        strategy: ChunkingStrategy,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Document:
        file_type = validate_upload(
            filename, content, content_type, self.settings.max_upload_size_bytes
        )
        if chunk_size > 10_000:
            raise ValueError("chunk_size cannot exceed 10000 characters.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        safe_filename = (filename or "").replace("\\", "/").rsplit("/", 1)[-1][:255]
        document = await self.repository.create(
            Document(
                filename=safe_filename,
                file_type=file_type,
                file_size=len(content),
                chunking_strategy=strategy.value,
                total_chunks=0,
                status=DocumentStatus.processing.value,
            )
        )
        try:
            text = self.extraction.extract(safe_filename, content)
            chunks = split_text(text, strategy, chunk_size, chunk_overlap)
            if not chunks:
                raise ValueError("The document produced no text chunks.")
            embeddings = await self.embeddings.embed_texts(chunks)
            await self.vectors.ensure_collection()
            await self.vectors.upsert_document(
                document.id, safe_filename, chunks, embeddings
            )
            document.total_chunks = len(chunks)
            document.status = DocumentStatus.ready.value
            return await self.repository.update(document)
        except Exception:
            try:
                await self.vectors.delete_document(document.id)
            except Exception:
                logger.exception(
                    "Failed to clean up Qdrant vectors after ingestion failure",
                    extra={"document_id": str(document.id)},
                )
            document.status = DocumentStatus.failed.value
            try:
                await self.repository.update(document)
            except Exception:
                logger.exception(
                    "Failed to mark document ingestion as failed",
                    extra={"document_id": str(document.id)},
                )
            raise

    async def get(self, document_id: UUID) -> Document:
        document = await self.repository.get(document_id)
        if not document:
            raise NotFoundError("Document")
        return document

    async def list(self, limit: int, offset: int) -> tuple[list[Document], int]:
        return await self.repository.list(limit=limit, offset=offset)

    async def delete(self, document_id: UUID) -> None:
        document = await self.repository.get(document_id)
        if not document:
            raise NotFoundError("Document")
        document.status = DocumentStatus.deleting.value
        await self.repository.update(document)
        try:
            await self.vectors.delete_document(document_id)
            await self.repository.delete(document)
        except Exception as exc:
            document.status = DocumentStatus.failed.value
            try:
                await self.repository.update(document)
            except Exception:
                logger.exception(
                    "Could not record failed document deletion",
                    extra={"document_id": str(document_id)},
                )
            if isinstance(exc, IntegrationUnavailableError):
                raise
            raise IntegrationUnavailableError(
                "Document deletion did not complete; retry the delete request."
            ) from exc
