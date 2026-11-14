from functools import lru_cache
from typing import Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings, get_settings
from app.db.session import get_db_session
from app.integrations.llm import LLMClient
from app.integrations.qdrant import QdrantVectorStore, create_qdrant_client
from app.integrations.redis_memory import RedisConversationMemory, create_redis_client
from app.repositories.booking_repository import BookingRepository
from app.repositories.document_repository import DocumentRepository
from app.services.booking_service import BookingService
from app.services.chat_service import ChatService
from app.services.embedding_service import EmbeddingService
from app.services.extraction_service import ExtractionService
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService


@lru_cache
def get_qdrant_client() -> Any:
    return create_qdrant_client(get_settings())


@lru_cache
def get_redis_client() -> Any:
    return create_redis_client(get_settings())


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings())


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient(get_settings())


def get_vector_store(
    settings: Settings = Depends(get_settings),
    client: Any = Depends(get_qdrant_client),
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client,
        settings.qdrant_collection,
        settings.embedding_dimension,
    )


def get_ingestion_service(
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    embeddings: EmbeddingService = Depends(get_embedding_service),
    vectors: QdrantVectorStore = Depends(get_vector_store),
) -> IngestionService:
    return IngestionService(
        DocumentRepository(session),
        ExtractionService(),
        embeddings,
        vectors,
        settings,
    )


def get_chat_service(
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    embeddings: EmbeddingService = Depends(get_embedding_service),
    vectors: QdrantVectorStore = Depends(get_vector_store),
    llm: LLMClient = Depends(get_llm_client),
) -> ChatService:
    memory = RedisConversationMemory(
        get_redis_client(),
        settings.chat_history_limit,
        settings.chat_session_ttl_seconds,
    )
    retrieval = RetrievalService(embeddings, vectors, settings.retrieval_top_k)
    booking_service = BookingService(BookingRepository(session))
    return ChatService(memory, retrieval, llm, booking_service)
