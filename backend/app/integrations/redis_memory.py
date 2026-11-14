import hashlib
import json
from typing import Any

from app.core.exceptions import IntegrationUnavailableError
from app.core.settings import Settings


class RedisConversationMemory:
    """Per-session conversation history and pending booking state."""

    def __init__(
        self,
        client: Any,
        history_limit: int,
        ttl_seconds: int,
    ) -> None:
        self.client = client
        self.history_limit = history_limit
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _key(session_id: str, suffix: str) -> str:
        session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return f"rag:session:{session_hash}:{suffix}"

    async def get_history(self, session_id: str) -> list[dict[str, str]]:
        try:
            encoded_messages = await self.client.lrange(
                self._key(session_id, "history"), 0, -1
            )
            return [json.loads(item) for item in encoded_messages]
        except Exception as exc:
            raise IntegrationUnavailableError("Conversation memory is unavailable.") from exc

    async def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        key = self._key(session_id, "history")
        messages = (
            json.dumps({"role": "user", "content": user_message}),
            json.dumps({"role": "assistant", "content": assistant_message}),
        )
        try:
            async with self.client.pipeline(transaction=True) as pipeline:
                pipeline.rpush(key, *messages)
                pipeline.ltrim(key, -2 * self.history_limit, -1)
                pipeline.expire(key, self.ttl_seconds)
                await pipeline.execute()
        except Exception as exc:
            raise IntegrationUnavailableError("Could not save conversation history.") from exc

    async def get_pending_booking(self, session_id: str) -> dict[str, str] | None:
        try:
            encoded = await self.client.get(self._key(session_id, "pending_booking"))
            return json.loads(encoded) if encoded else None
        except Exception as exc:
            raise IntegrationUnavailableError("Pending booking state is unavailable.") from exc

    async def set_pending_booking(
        self, session_id: str, booking: dict[str, str]
    ) -> None:
        try:
            await self.client.set(
                self._key(session_id, "pending_booking"),
                json.dumps(booking),
                ex=self.ttl_seconds,
            )
        except Exception as exc:
            raise IntegrationUnavailableError("Could not save pending booking details.") from exc

    async def clear_pending_booking(self, session_id: str) -> None:
        try:
            await self.client.delete(self._key(session_id, "pending_booking"))
        except Exception as exc:
            raise IntegrationUnavailableError("Could not clear pending booking details.") from exc


def create_redis_client(settings: Settings) -> Any:
    from redis.asyncio import Redis

    return Redis.from_url(settings.redis_url, decode_responses=True)
