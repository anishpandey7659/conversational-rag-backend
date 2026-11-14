from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from qdrant_client import AsyncQdrantClient, models

from app.core.exceptions import IntegrationUnavailableError
from app.core.settings import Settings


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: UUID
    filename: str
    chunk_index: int
    text: str
    score: float


class QdrantVectorStore:
    """Qdrant operations used by ingestion, retrieval, and deletion."""

    def __init__(
        self,
        client: Any,
        collection_name: str,
        vector_dimension: int,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension

    async def ensure_collection(self) -> None:
        try:
            exists = await self.client.collection_exists(self.collection_name)
            if not exists:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as exc:
            raise IntegrationUnavailableError("Qdrant collection is unavailable.") from exc

    async def upsert_document(
        self,
        document_id: UUID,
        filename: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding.")
        points = [
            models.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "document_id": str(document_id),
                    "filename": filename,
                    "chunk_index": index,
                    "text": chunk,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        try:
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise IntegrationUnavailableError("Could not store document vectors in Qdrant.") from exc

    async def delete_document(self, document_id: UUID) -> None:
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=str(document_id)),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except Exception as exc:
            raise IntegrationUnavailableError(
                "Could not remove document vectors from Qdrant."
            ) from exc

    async def search(
        self, vector: list[float], limit: int
    ) -> list[RetrievedChunk]:
        try:
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            raise IntegrationUnavailableError("Qdrant vector search failed.") from exc

        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            try:
                results.append(
                    RetrievedChunk(
                        document_id=UUID(str(payload["document_id"])),
                        filename=str(payload["filename"]),
                        chunk_index=int(payload["chunk_index"]),
                        text=str(payload["text"]),
                        score=float(point.score),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return results


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    api_key = (
        settings.qdrant_api_key.get_secret_value().strip()
        if settings.qdrant_api_key and settings.qdrant_api_key.get_secret_value().strip()
        else None
    )
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=api_key,
        check_compatibility=False,
    )
