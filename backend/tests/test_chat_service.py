from datetime import date, datetime, time, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.integrations.llm import LLMResult, ToolCall
from app.integrations.qdrant import RetrievedChunk
from app.services.chat_service import ChatService


class FakeMemory:
    def __init__(self) -> None:
        self.history: dict[str, list[dict[str, str]]] = {}
        self.pending: dict[str, dict[str, str]] = {}

    async def get_history(self, session_id: str):
        return list(self.history.get(session_id, []))

    async def append_turn(self, session_id: str, user_message: str, assistant_message: str):
        self.history.setdefault(session_id, []).extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )

    async def get_pending_booking(self, session_id: str):
        return self.pending.get(session_id)

    async def set_pending_booking(self, session_id: str, booking: dict[str, str]):
        self.pending[session_id] = booking

    async def clear_pending_booking(self, session_id: str):
        self.pending.pop(session_id, None)


class FakeRetrieval:
    def __init__(self) -> None:
        self.chunks = [
            RetrievedChunk(
                document_id=uuid4(),
                filename="findings.pdf",
                chunk_index=2,
                text="The study found a 12 percent improvement.",
                score=0.91,
            )
        ]

    async def retrieve(self, _: str):
        return self.chunks


class FakeLLM:
    def __init__(self, results: list[LLMResult]) -> None:
        self.results = results
        self.messages = []

    async def generate(self, messages):
        self.messages.append(messages)
        return self.results.pop(0)


class FakeBookingService:
    def __init__(self) -> None:
        self.booked = []

    async def book_interview(self, name, email, date, time, booking_id=None):
        from app.schemas.booking import BookingDetails

        details = BookingDetails(name=name, email=email, date=date, time=time)
        self.booked.append(details)
        return SimpleNamespace(
            id=booking_id or uuid4(),
            name=details.name,
            email=str(details.email),
            interview_date=details.date,
            interview_time=details.time,
            status="confirmed",
            created_at=datetime.now(timezone.utc),
        )


def make_service(results: list[LLMResult]):
    memory = FakeMemory()
    llm = FakeLLM(results)
    booking = FakeBookingService()
    service = ChatService(memory, FakeRetrieval(), llm, booking)
    return service, memory, llm, booking


@pytest.mark.asyncio
async def test_multiturn_follow_up_uses_history_and_returns_sources() -> None:
    service, memory, llm, _ = make_service(
        [LLMResult("The study found a 12 percent improvement."), LLMResult("It was measured against the control group.")]
    )
    first = await service.chat("session-a", "What were the findings?")
    second = await service.chat("session-a", "What was it compared against?")

    assert first.sources[0].filename == "findings.pdf"
    assert second.answer.startswith("It was measured")
    assert any(
        item["content"] == "What were the findings?"
        for item in llm.messages[1]
    )
    assert len(memory.history["session-a"]) == 4


@pytest.mark.asyncio
async def test_booking_tool_stages_details_then_requires_explicit_confirmation() -> None:
    service, memory, _, booking_service = make_service(
        [
            LLMResult(
                "",
                ToolCall(
                    name="book_interview",
                    arguments={
                        "name": "Anish Rai",
                        "email": "anish@example.com",
                        "date": "2099-11-20",
                        "time": "11:00",
                    },
                ),
            )
        ]
    )
    staged = await service.chat("booking-session", "I want to book an interview.")
    assert staged.pending_booking is not None
    assert not booking_service.booked
    assert "No booking has been submitted yet" in staged.answer
    assert memory.pending["booking-session"]["booking_id"]

    confirmed = await service.chat("booking-session", "yes, confirm")
    assert confirmed.booking is not None
    assert confirmed.booking.status == "confirmed"
    assert len(booking_service.booked) == 1
    assert "confirmed" in confirmed.answer.lower()
    assert "booking-session" not in memory.pending


@pytest.mark.asyncio
async def test_invalid_booking_tool_data_is_not_staged() -> None:
    service, memory, _, booking_service = make_service(
        [
            LLMResult(
                "",
                ToolCall(
                    name="book_interview",
                    arguments={
                        "name": "Anish",
                        "email": "invalid-email",
                        "date": "2099-11-20",
                        "time": "25:90",
                    },
                ),
            )
        ]
    )
    response = await service.chat("bad-booking", "Book me in.")
    assert "could not validate" in response.answer.lower()
    assert "bad-booking" not in memory.pending
    assert not booking_service.booked


@pytest.mark.asyncio
async def test_model_cannot_claim_booking_success_without_persistence() -> None:
    service, _, _, booking_service = make_service(
        [LLMResult("Your interview has been confirmed and booked.")]
    )
    response = await service.chat("no-claim", "Did you book it?")
    assert "have not submitted or confirmed" in response.answer
    assert not booking_service.booked


@pytest.mark.asyncio
async def test_pending_booking_can_be_cancelled_without_persistence() -> None:
    service, memory, _, booking_service = make_service([])
    memory.pending["cancel-session"] = {
        "name": "Anish Rai",
        "email": "anish@example.com",
        "date": "2099-11-20",
        "time": "11:00",
    }
    response = await service.chat("cancel-session", "cancel")
    assert "No booking was submitted" in response.answer
    assert not booking_service.booked
    assert "cancel-session" not in memory.pending


@pytest.mark.asyncio
async def test_missing_booking_fields_are_collected_without_saving() -> None:
    service, memory, _, booking_service = make_service(
        [
            LLMResult(
                "",
                ToolCall(
                    name="book_interview",
                    arguments={"name": "Anish Rai", "email": "anish@example.com"},
                ),
            )
        ]
    )
    response = await service.chat("collect-session", "I want an interview.")
    assert "preferred date" in response.answer
    assert "preferred time" in response.answer
    assert not booking_service.booked
    assert "collect-session" not in memory.pending
