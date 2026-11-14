import logging
import re
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.core.exceptions import IntegrationUnavailableError
from app.integrations.llm import LLMClient, ToolCall
from app.integrations.redis_memory import RedisConversationMemory
from app.integrations.qdrant import RetrievedChunk
from app.schemas.booking import BookingDetails, BookingRead
from app.schemas.chat import ChatResponse, SourceReference
from app.services.booking_service import BookingService
from app.services.prompt_service import build_messages
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

_CONFIRMATIONS = {
    "yes",
    "yes i confirm",
    "yes confirm",
    "yes please",
    "confirm",
    "please confirm",
    "i confirm",
    "confirmed",
    "book it",
    "please book it",
    "that is correct",
    "that's correct",
    "that looks correct",
    "looks good",
    "please go ahead",
    "yes that works",
    "yes that is correct",
    "go ahead",
    "go ahead and book it",
}
_REJECTIONS = {"no", "no thanks", "cancel", "cancel it", "do not book", "don't book"}
_SUCCESS_CLAIM = re.compile(
    r"\b(?:booked|scheduled|reserved)\b.{0,100}\b(?:interview|booking|time)\b|"
    r"\b(?:interview|booking)\b.{0,100}\b(?:booked|scheduled|reserved|confirmed|"
    r"submitted|saved)\b",
    re.IGNORECASE,
)


def _normalized_confirmation(message: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", "", message.casefold()).split())


def is_explicit_confirmation(message: str) -> bool:
    return _normalized_confirmation(message) in _CONFIRMATIONS


def is_explicit_rejection(message: str) -> bool:
    return _normalized_confirmation(message) in _REJECTIONS


def _normalize_tool_details(arguments: dict[str, Any]) -> BookingDetails:
    payload = dict(arguments)
    if "interview_date" in payload and "date" not in payload:
        payload["date"] = payload.pop("interview_date")
    if "interview_time" in payload and "time" not in payload:
        payload["time"] = payload.pop("interview_time")
    if isinstance(payload.get("time"), str):
        raw_time = payload["time"].strip()
        for format_string in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I %p"):
            try:
                payload["time"] = datetime.strptime(raw_time, format_string).time()
                break
            except ValueError:
                continue
    try:
        return BookingDetails.model_validate(payload)
    except ValidationError as exc:
        raise exc


def _missing_fields(tool_call: ToolCall) -> list[str]:
    labels = {
        "name": "your full name",
        "email": "your email address",
        "date": "a preferred date in YYYY-MM-DD format",
        "time": "a preferred time such as 11:00",
    }
    return [labels[key] for key in labels if not tool_call.arguments.get(key)]


def _sources(chunks: list[RetrievedChunk]) -> list[SourceReference]:
    unique: dict[tuple[str, int], SourceReference] = {}
    for chunk in chunks:
        key = (str(chunk.document_id), chunk.chunk_index)
        unique[key] = SourceReference(
            document_id=chunk.document_id,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            score=chunk.score,
        )
    return list(unique.values())


class ChatService:
    """Build a grounded response and enforce confirmed-only booking persistence."""

    def __init__(
        self,
        memory: RedisConversationMemory,
        retrieval: RetrievalService,
        llm: LLMClient,
        booking_service: BookingService,
    ) -> None:
        self.memory = memory
        self.retrieval = retrieval
        self.llm = llm
        self.booking_service = booking_service

    async def chat(self, session_id: str, message: str) -> ChatResponse:
        pending = await self.memory.get_pending_booking(session_id)

        if pending and is_explicit_confirmation(message):
            try:
                details = BookingDetails.model_validate(pending)
                booking = await self.booking_service.book_interview(
                    name=details.name,
                    email=str(details.email),
                    date=details.date.isoformat(),
                    time=details.time.strftime("%H:%M"),
                    booking_id=UUID(pending["booking_id"]) if pending.get("booking_id") else None,
                )
            except ValidationError as exc:
                await self.memory.clear_pending_booking(session_id)
                raise IntegrationUnavailableError(
                    "The staged booking details are no longer valid. Please provide them again."
                ) from exc
            except Exception:
                logger.exception("Confirmed booking could not be persisted")
                raise
            answer = (
                f"Your interview booking is confirmed for {booking.interview_date.isoformat()} "
                f"at {booking.interview_time.strftime('%H:%M')}. A confirmation was saved."
            )
            await self.memory.append_turn(session_id, message, answer)
            await self.memory.clear_pending_booking(session_id)
            return ChatResponse(
                session_id=session_id,
                answer=answer,
                sources=[],
                booking=BookingRead.model_validate(booking),
            )

        if pending and is_explicit_rejection(message):
            await self.memory.clear_pending_booking(session_id)
            answer = "No booking was submitted. The staged interview details were cancelled."
            await self.memory.append_turn(session_id, message, answer)
            return ChatResponse(session_id=session_id, answer=answer, sources=[])

        history = await self.memory.get_history(session_id)
        chunks = await self.retrieval.retrieve(message)
        messages = build_messages(history, message, chunks, pending)
        result = await self.llm.generate(messages)

        pending_response: dict[str, str] | None = None
        if result.tool_call:
            if result.tool_call.name != "book_interview":
                raise IntegrationUnavailableError(
                    f"The LLM requested unsupported tool '{result.tool_call.name}'."
                )
            missing = _missing_fields(result.tool_call)
            if missing:
                if pending:
                    await self.memory.clear_pending_booking(session_id)
                answer = "To prepare the booking, please provide " + ", ".join(missing) + "."
            else:
                try:
                    details = _normalize_tool_details(result.tool_call.arguments)
                except ValidationError:
                    if pending:
                        await self.memory.clear_pending_booking(session_id)
                    answer = (
                        "I could not validate those booking details. Please check the "
                        "email, use a future date in YYYY-MM-DD format, and provide a "
                        "valid time such as 11:00."
                    )
                else:
                    pending_response = {
                        "name": details.name,
                        "email": str(details.email),
                        "date": details.date.isoformat(),
                        "time": details.time.strftime("%H:%M"),
                    }
                    pending_state = {
                        **pending_response,
                        "booking_id": str(uuid4()),
                    }
                    await self.memory.set_pending_booking(session_id, pending_state)
                    answer = (
                        "Please confirm these details before I submit the booking: "
                        f"{details.name}, {details.email}, {details.date.isoformat()} "
                        f"at {details.time.strftime('%H:%M')}. Reply “yes, confirm” to "
                        "save it, or tell me what to change. No booking has been submitted yet."
                    )
        else:
            if pending:
                await self.memory.clear_pending_booking(session_id)
            answer = result.text.strip()
            if not answer:
                answer = "I could not generate a response. Please try again."
            if pending:
                answer += (
                    "\n\nThe earlier booking details were not submitted. To book, "
                    "please ask again and confirm the displayed details."
                )
            if _SUCCESS_CLAIM.search(answer):
                answer = (
                    "I have not submitted or confirmed a booking. If you want to book "
                    "an interview, share your name, email, preferred date, and time."
                )

        await self.memory.append_turn(session_id, message, answer)
        return ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=_sources(chunks),
            pending_booking=pending_response,
        )
