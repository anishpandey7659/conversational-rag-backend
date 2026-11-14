import pytest

from app.integrations.redis_memory import RedisConversationMemory


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    def rpush(self, key, *values):
        self.commands.append(("rpush", key, values))
        return self

    def ltrim(self, key, start, end):
        self.commands.append(("ltrim", key, start, end))
        return self

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))
        return self

    async def execute(self):
        for command in self.commands:
            if command[0] == "rpush":
                _, key, values = command
                self.client.lists.setdefault(key, []).extend(values)
            elif command[0] == "ltrim":
                _, key, start, end = command
                items = self.client.lists.get(key, [])
                normalized_start = start if start >= 0 else max(len(items) + start, 0)
                normalized_end = end if end >= 0 else len(items) + end
                self.client.lists[key] = items[normalized_start : normalized_end + 1]
        return []


class FakeRedis:
    def __init__(self):
        self.lists = {}
        self.values = {}

    async def lrange(self, key, start, end):
        values = self.lists.get(key, [])
        normalized_start = start if start >= 0 else max(len(values) + start, 0)
        normalized_end = end if end >= 0 else len(values) + end
        return values[normalized_start : normalized_end + 1]

    def pipeline(self, transaction=True):
        assert transaction
        return FakePipeline(self)

    async def set(self, key, value, ex=None):
        self.values[key] = value

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.values.pop(key, None)
        self.lists.pop(key, None)


@pytest.mark.asyncio
async def test_redis_memory_limits_history_and_isolates_sessions() -> None:
    redis = FakeRedis()
    memory = RedisConversationMemory(redis, history_limit=2, ttl_seconds=120)
    await memory.append_turn("session-a", "first", "answer one")
    await memory.append_turn("session-a", "second", "answer two")
    await memory.append_turn("session-a", "third", "answer three")
    await memory.append_turn("session-b", "other user", "separate answer")

    history_a = await memory.get_history("session-a")
    history_b = await memory.get_history("session-b")
    assert len(history_a) == 4
    assert history_a[0]["content"] == "second"
    assert len(history_b) == 2
    assert history_b[0]["content"] == "other user"


@pytest.mark.asyncio
async def test_pending_booking_is_session_scoped_and_clearable() -> None:
    memory = RedisConversationMemory(FakeRedis(), history_limit=4, ttl_seconds=120)
    details = {"name": "Anish Rai", "date": "2099-11-20"}
    await memory.set_pending_booking("session-a", details)

    assert await memory.get_pending_booking("session-a") == details
    assert await memory.get_pending_booking("session-b") is None
    await memory.clear_pending_booking("session-a")
    assert await memory.get_pending_booking("session-a") is None
