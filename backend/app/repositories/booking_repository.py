from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DataIntegrityError
from app.models.booking import Booking
from app.schemas.booking import BookingDetails


class BookingRepository:
    """Persistence operations for confirmed interview bookings."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self, details: BookingDetails, booking_id: UUID | None = None
    ) -> Booking:
        if booking_id:
            existing = await self.session.get(Booking, booking_id)
            if existing:
                return existing
        booking = Booking(
            id=booking_id or uuid4(),
            name=details.name,
            email=str(details.email),
            interview_date=details.date,
            interview_time=details.time,
            status="confirmed",
        )
        self.session.add(booking)
        try:
            await self.session.commit()
            await self.session.refresh(booking)
        except IntegrityError as exc:
            await self.session.rollback()
            if booking_id:
                existing = await self.session.scalar(
                    select(Booking).where(Booking.id == booking_id)
                )
                if existing:
                    return existing
            raise DataIntegrityError("The booking could not be saved.") from exc
        return booking
