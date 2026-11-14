# Conversational RAG Backend

A backend-only service for document ingestion, grounded multi-turn chat, and
interview booking. It uses FastAPI, PostgreSQL, Redis, Qdrant, local Sentence
Transformers embeddings, and an OpenAI-compatible LLM API.

## Quick start

From the repository root:

```bash
cp backend/.env.example .env
# Add OPENAI_API_KEY and/or GROQ_API_KEY to .env for live chat.
docker compose up --build
```

The API is available at `http://localhost:8000`. Interactive API documentation is
at `http://localhost:8000/docs`; health endpoints are `/healthz` and `/readyz`.

Set `LLM_PROVIDER=openai` or `LLM_PROVIDER=groq` to choose the active provider.
Provider keys are optional for running the service and test suite, but required for
live chat requests. Do not commit `.env`; use Replit Secrets when running in Replit.

## Main endpoints

- `POST /api/v1/documents/upload` — ingest a PDF or UTF-8 TXT file.
- `GET /api/v1/documents` — list document metadata.
- `GET /api/v1/documents/{document_id}` — read a document record.
- `DELETE /api/v1/documents/{document_id}` — remove its vectors and metadata.
- `POST /api/v1/chat` — ask questions or start an interview booking conversation.

Interview details are saved only after the user explicitly confirms the details
shown by the assistant.

## Tests

```bash
cd backend
pip install -r requirements.txt
pytest -q
```

The tests mock external AI and storage services, so they do not require paid API
calls or running PostgreSQL, Redis, or Qdrant instances.

## Project guide

See [`backend/README.md`](backend/README.md) for the architecture, environment
variables, API examples, storage details, and design limitations.
