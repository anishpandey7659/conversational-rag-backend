import json
from dataclasses import dataclass
from typing import Any

from app.core.exceptions import IntegrationUnavailableError
from app.core.settings import Settings

BOOK_INTERVIEW_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "book_interview",
        "description": (
            "Prepare an interview booking for user confirmation. Call only after "
            "collecting all fields. The application will not submit it until the user "
            "explicitly confirms the displayed details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Candidate's full name"},
                "email": {"type": "string", "description": "Candidate's email address"},
                "date": {
                    "type": "string",
                    "description": "Interview date in YYYY-MM-DD format",
                },
                "time": {
                    "type": "string",
                    "description": "Interview time in 24-hour HH:MM format",
                },
            },
            "required": ["name", "email", "date", "time"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResult:
    text: str
    tool_call: ToolCall | None = None


class LLMClient:
    """OpenAI-compatible chat completion adapter for OpenAI and Groq."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        configured_key = self.settings.resolved_llm_api_key
        api_key = (
            configured_key.get_secret_value().strip()
            if configured_key
            else ""
        )
        if not api_key:
            raise IntegrationUnavailableError(
                "No API key is configured for the selected LLM provider. Set "
                "OPENAI_API_KEY, GROQ_API_KEY, or the shared LLM_API_KEY fallback."
            )
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.settings.resolved_llm_base_url,
            )
        except Exception as exc:
            raise IntegrationUnavailableError("The LLM client could not be initialized.") from exc
        return self._client

    async def generate(self, messages: list[dict[str, str]]) -> LLMResult:
        try:
            completion = await self._get_client().chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                tools=[BOOK_INTERVIEW_TOOL],
                tool_choice="auto",
                parallel_tool_calls=False,
                temperature=0.2,
            )
        except IntegrationUnavailableError:
            raise
        except Exception as exc:
            raise IntegrationUnavailableError("The configured LLM request failed.") from exc

        choice = completion.choices[0]
        message = choice.message
        if message.tool_calls:
            call = message.tool_calls[0]
            try:
                arguments = json.loads(call.function.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raise IntegrationUnavailableError(
                    "The LLM returned invalid booking tool arguments."
                ) from exc
            return LLMResult(
                text=message.content or "",
                tool_call=ToolCall(name=call.function.name, arguments=arguments),
            )
        return LLMResult(text=message.content or "")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
