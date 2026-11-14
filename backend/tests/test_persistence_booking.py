from datetime import date, datetime, time, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.models.document import Document
from app.repositories.booking_repository import BookingRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.booking import BookingDetails


@pytest.mark.asyncio
async def test_document_metadata_persists_and_can_be_removed() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        repository = DocumentRepository(session)
        document = await repository.create(
            Document(
                filename="manual.txt",
                file_type="txt",
                file_size=50,
                chunking_strategy="fixed",
                total_chunks=2,
                status="ready",
            )
        )
        stored = await repository.get(document.id)
        page, total = await repository.list(limit=10, offset=0)
        assert stored is not None
        assert stored.filename == "manual.txt"
        assert total == 1
        assert page[0].id == document.id
        await repository.delete(stored)
        assert await repository.get(document.id) is None

    await engine.dispose()


@pytest.mark.asyncio
async def test_booking_is_persisted_only_as_confirmed_record() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        details = BookingDetails(
            name="  Anish   Rai ",
            email="ANISH@example.com",
            date=date(2099, 11, 20),
            time=time(11, 0),
        )
        booking = await BookingRepository(session).create(details)
        assert booking.name == "Anish Rai"
        assert booking.email == "anish@example.com"
        assert booking.status == "confirmed"
        assert booking.interview_date == date(2099, 11, 20)
        assert booking.created_at is not None

    await engine.dispose()


@pytest.mark.asyncio
async def test_confirmed_booking_retry_is_idempotent() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    booking_id = uuid4()
    details = BookingDetails(
        name="Anish Rai",
        email="anish@example.com",
        date=date(2099, 11, 20),
        time=time(11, 0),
    )

    async with factory() as session:
        repository = BookingRepository(session)
        first = await repository.create(details, booking_id=booking_id)
        retry = await repository.create(details, booking_id=booking_id)
        assert first.id == retry.id == booking_id

    await engine.dispose()


def test_invalid_booking_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        BookingDetails(
            name="",
            email="not-an-email",
            date=date(2000, 1, 1),
            time=time(11, 0),
        )
