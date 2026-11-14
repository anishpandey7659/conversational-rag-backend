from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategy(StrEnum):
    fixed = "fixed"
    recursive = "recursive"


class DocumentStatus(StrEnum):
    processing = "processing"
    ready = "ready"
    failed = "failed"
    deleting = "deleting"


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    file_type: str
    file_size: int
    chunking_strategy: ChunkingStrategy
    total_chunks: int
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime


class DocumentUploadRead(BaseModel):
    status: DocumentStatus
    document_id: UUID
    filename: str
    chunking_strategy: ChunkingStrategy
    chunk_count: int


class DocumentListRead(BaseModel):
    items: list[DocumentRead]
    total: int
    limit: int
    offset: int


class DocumentListQuery(BaseModel):
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
