from app.integrations.qdrant import QdrantVectorStore, RetrievedChunk
from app.services.embedding_service import EmbeddingService


class RetrievalService:
    """Embed a query and retrieve its nearest document chunks."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        vectors: QdrantVectorStore,
        top_k: int,
    ) -> None:
        self.embeddings = embeddings
        self.vectors = vectors
        self.top_k = top_k

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        query_vector = await self.embeddings.embed_query(query)
        return await self.vectors.search(query_vector, self.top_k)
