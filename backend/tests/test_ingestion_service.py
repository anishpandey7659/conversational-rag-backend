from uuid import uuid4

import pytest

from app.core.settings import Settings
from app.models.document import Document
from app.schemas.document import ChunkingStrategy
from app.services.extraction_service import ExtractionService
from app.services.ingestion_service import IngestionService


class FakeDocumentRepository:
    def __init__(self) -> None:
        self.document = None

    async def create(self, document: Document) -> Document:
        document.id = uuid4()
        self.document = document
        return document

    async def update(self, document: Document) -> Document:
        self.document = document
        return document


class FakeEmbeddings:
    async def embed_texts(self, chunks: list[str]) -> list[list[float]]:
        return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(chunks)]


class FailingEmbeddings:
    async def embed_texts(self, _: list[str]) -> list[list[float]]:
        raise RuntimeError("mock embedding failure")


class FakeVectors:
    def __init__(self) -> None:
        self.deleted = False
        self.chunks = []

    async def ensure_collection(self) -> None:
        return None

    async def upsert_document(self, _, filename, chunks, vectors) -> None:
        assert filename
        assert len(chunks) == len(vectors)
        self.chunks = chunks

    async def delete_document(self, _) -> None:
        self.deleted = True
        self.chunks = []


@pytest.mark.asyncio
async def test_ingestion_persists_metadata_and_vectors() -> None:
    repository = FakeDocumentRepository()
    vectors = FakeVectors()
    service = IngestionService(
        repository,
        extraction=ExtractionService(),
        embeddings=FakeEmbeddings(),
        vectors=vectors,
        settings=Settings(_env_file=None, embedding_dimension=3),
    )
    document = await service.ingest(
        "guide.txt",
        "text/plain",
        b"Chunk one. Chunk two. Chunk three.",
        ChunkingStrategy.fixed,
        chunk_size=12,
        chunk_overlap=2,
    )
    assert document.status == "ready"
    assert document.total_chunks == len(vectors.chunks)
    assert vectors.chunks


@pytest.mark.asyncio
async def test_partial_ingestion_failure_cleans_vectors_and_marks_failed() -> None:
    repository = FakeDocumentRepository()
    vectors = FakeVectors()
    service = IngestionService(
        repository,
        extraction=ExtractionService(),
        embeddings=FailingEmbeddings(),
        vectors=vectors,
        settings=Settings(_env_file=None, embedding_dimension=3),
    )
    with pytest.raises(RuntimeError, match="embedding failure"):
        await service.ingest(
            "guide.txt",
            "text/plain",
            b"Readable text for embedding.",
            ChunkingStrategy.recursive,
            chunk_size=20,
            chunk_overlap=3,
        )
    assert repository.document.status == "failed"
    assert vectors.deleted
