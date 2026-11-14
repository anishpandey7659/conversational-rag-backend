from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.exceptions import IntegrationUnavailableError
from app.core.settings import Settings
from app.integrations.llm import LLMClient
from app.integrations.qdrant import QdrantVectorStore
from app.services.embedding_service import EmbeddingService


class Matrix(list):
    def tolist(self):
        return list(self)


class FakeEmbeddingModel:
    def encode(self, texts, **_: object) -> Matrix:
        return Matrix([[float(len(text)), 0.25, 0.5] for text in texts])


class FakeQdrant:
    def __init__(self) -> None:
        self.collection = False
        self.points = []
        self.deleted = False

    async def collection_exists(self, _: str) -> bool:
        return self.collection

    async def create_collection(self, **_: object) -> None:
        self.collection = True

    async def upsert(self, **kwargs: object) -> None:
        self.points.extend(kwargs["points"])

    async def delete(self, **_: object) -> None:
        self.deleted = True
        self.points.clear()

    async def query_points(self, **_: object) -> SimpleNamespace:
        payload = self.points[0].payload
        return SimpleNamespace(
            points=[SimpleNamespace(payload=payload, score=0.92)]
        )


@pytest.mark.asyncio
async def test_embedding_generation_uses_local_model() -> None:
    settings = Settings(_env_file=None, embedding_dimension=3)
    service = EmbeddingService(settings, model=FakeEmbeddingModel())
    assert await service.embed_texts(["one", "two words"]) == [
        [3.0, 0.25, 0.5],
        [9.0, 0.25, 0.5],
    ]


@pytest.mark.asyncio
async def test_chat_reports_missing_llm_key_without_making_a_request() -> None:
    client = LLMClient(Settings(_env_file=None, llm_api_key=None))
    with pytest.raises(IntegrationUnavailableError, match="LLM_API_KEY"):
        await client.generate([{"role": "user", "content": "hello"}])


def test_llm_provider_selects_its_own_key_with_shared_fallback() -> None:
    openai_settings = Settings(
        _env_file=None,
        llm_provider="openai",
        openai_api_key="openai-example",
        groq_api_key="groq-example",
    )
    groq_settings = Settings(
        _env_file=None,
        llm_provider="groq",
        openai_api_key="openai-example",
        groq_api_key="groq-example",
    )
    assert openai_settings.resolved_llm_api_key.get_secret_value() == "openai-example"
    assert groq_settings.resolved_llm_api_key.get_secret_value() == "groq-example"

    fallback = Settings(
        _env_file=None,
        llm_provider="groq",
        llm_api_key="shared-example",
    )
    assert fallback.resolved_llm_api_key.get_secret_value() == "shared-example"


@pytest.mark.asyncio
async def test_qdrant_collection_insert_search_and_delete() -> None:
    client = FakeQdrant()
    store = QdrantVectorStore(client, "documents", 3)
    document_id = uuid4()

    await store.ensure_collection()
    await store.upsert_document(document_id, "guide.txt", ["source text"], [[1, 0, 0]])
    found = await store.search([1, 0, 0], limit=2)
    assert found[0].document_id == document_id
    assert found[0].filename == "guide.txt"
    assert found[0].text == "source text"

    await store.delete_document(document_id)
    assert client.deleted
    assert client.points == []
