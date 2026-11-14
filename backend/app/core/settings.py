from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/ragdb"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"

    llm_provider: str = "openai"
    llm_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, gt=0)
    max_upload_size_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    chat_history_limit: int = Field(default=12, ge=1, le=100)
    chat_session_ttl_seconds: int = Field(default=86_400, gt=0)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    auto_create_tables: bool = True

    @model_validator(mode="after")
    def validate_provider(self) -> "Settings":
        provider = self.llm_provider.lower()
        if provider not in {"openai", "groq"}:
            raise ValueError("LLM_PROVIDER must be either 'openai' or 'groq'.")
        self.llm_provider = provider
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace(
                "postgres://", "postgresql+asyncpg://", 1
            )
        elif self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        return self

    @property
    def resolved_llm_base_url(self) -> str | None:
        if self.llm_base_url:
            return self.llm_base_url
        if self.llm_provider == "groq":
            return "https://api.groq.com/openai/v1"
        return None

    @property
    def resolved_llm_api_key(self) -> SecretStr | None:
        provider_key = (
            self.openai_api_key if self.llm_provider == "openai" else self.groq_api_key
        )
        if provider_key and provider_key.get_secret_value().strip():
            return provider_key
        return self.llm_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
