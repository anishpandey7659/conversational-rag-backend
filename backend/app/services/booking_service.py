from uuid import UUID

from app.models.booking import Booking
from app.repositories.booking_repository import BookingRepository
from app.schemas.booking import BookingDetails


class BookingService:
    """Validate and persist confirmed interview bookings."""

    def __init__(self, repository: BookingRepository) -> None:
        self.repository = repository

    async def book_interview(
        self,
        name: str,
        email: str,
        date: str,
        time: str,
        booking_id: UUID | None = None,
    ) -> Booking:
        """Validate the tool arguments and persist one confirmed interview."""
        details = BookingDetails.model_validate(
            {"name": name, "email": email, "date": date, "time": time}
        )
        return await self.repository.create(details, booking_id=booking_id)
