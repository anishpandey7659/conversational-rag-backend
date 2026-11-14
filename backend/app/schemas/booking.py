from datetime import date, datetime, time, timezone
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class BookingDetails(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    date: date
    time: time

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Name cannot be blank.")
        return normalized

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> EmailStr:
        return str(value).strip().lower()

    @field_validator("date")
    @classmethod
    def require_future_date(cls, value: date) -> date:
        if value < datetime.now(timezone.utc).date():
            raise ValueError("Interview date must not be in the past.")
        return value


class BookingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    email: EmailStr
    interview_date: date
    interview_time: time
    status: str
    created_at: datetime
