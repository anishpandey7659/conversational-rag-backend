from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.schemas.booking import BookingRead


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=12_000)

    @field_validator("session_id", "message")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be blank.")
        return value


class SourceReference(BaseModel):
    document_id: UUID
    filename: str
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceReference]
    pending_booking: dict[str, str] | None = None
    booking: BookingRead | None = None
