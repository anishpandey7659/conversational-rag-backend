# Conversational RAG Backend

A backend-only FastAPI service for uploading PDF/TXT documents, retrieving grounded
answers across multi-turn conversations, and collecting confirmed interview bookings.
It uses PostgreSQL for metadata and bookings, Redis for session memory, Qdrant for
vectors, local Sentence Transformers embeddings, and an OpenAI-compatible LLM endpoint.

## Architecture

```text
HTTP routes (FastAPI + Pydantic)
  ├── Ingestion service
  │     ├── PDF/TXT extraction → chunking → local embeddings
  │     ├── PostgreSQL document metadata
  │     └── Qdrant chunk vectors
  └── Chat service
        ├── Redis session history and pending booking state
        ├── query embedding → Qdrant similarity search
        ├── explicit prompt construction → OpenAI/Groq-compatible LLM
        └── confirmed booking → PostgreSQL
```

Routes validate inputs and translate expected application errors into stable JSON
responses. Services coordinate business behavior; repositories own PostgreSQL access;
integration adapters isolate Redis, Qdrant, embeddings, and LLM calls. The RAG prompt,
retrieval, and generation are separate application steps—there is no prebuilt
end-to-end retrieval chain.

### Key decisions and limitations

- Sentence Transformers runs locally; model loading is lazy so health checks and unit
  tests do not need to download a model. The first embedding request may download the
  configured model.
- `LLM_PROVIDER` selects OpenAI or Groq through the OpenAI-compatible API. You can
  configure both provider-specific keys and switch between them without replacing a
  shared key. No provider credential is committed. If the selected provider has no
  key, chat returns a clear configuration error while ingestion, health checks, and
  mocked tests remain usable.
- Booking details are staged in Redis, shown back to the user, and saved only after
  an explicit confirmation. A stable booking UUID makes confirmation retries
  idempotent. The app does not connect to a calendar or claim time availability.
- PostgreSQL metadata and Qdrant vectors cannot share one transaction. Ingestion
  records a failure state and attempts vector cleanup if a later step fails. Deletion
  is idempotent and can be retried after a partial failure.
- Tables are created at startup by default to keep the assignment easy to run.
  Production deployments should use versioned migrations and managed credentials.
- Compose credentials are local-development defaults only; replace them before any
  non-local deployment.

## Stack

- Python 3.11+
- FastAPI, Pydantic v2, SQLAlchemy 2 async
- PostgreSQL / asyncpg
- Redis asyncio client
- Qdrant async client
- Sentence Transformers and PyMuPDF
- OpenAI SDK for OpenAI or Groq-compatible inference
- Docker Compose and pytest

## Run with Docker Compose

From the repository root:

```bash
cp backend/.env.example .env
# Set OPENAI_API_KEY and/or GROQ_API_KEY in .env for live chat inference.
docker compose up --build
```

The first image build and first embedding request can take several minutes because
the CPU PyTorch package and language model are large. The services expose:

- API: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`
- Liveness: `http://localhost:8000/healthz`
- Readiness: `http://localhost:8000/readyz`

Compose creates persistent volumes for PostgreSQL, Redis, Qdrant, and the model cache.
The default database password is deliberately only for local development. Do not use
it outside a local machine.

To stop services without removing data:

```bash
docker compose down
```

To remove all persisted service data (including documents, bookings, chat memory, and
vectors), explicitly run:

```bash
docker compose down -v
```

## Run locally without Docker

Start PostgreSQL, Redis, and Qdrant separately, then from `backend/`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The provided database URL assumes a local PostgreSQL database named `ragdb` with
user/password `rag`. Update `DATABASE_URL`, `REDIS_URL`, and `QDRANT_URL` for your
local service configuration. Use `LLM_PROVIDER=groq` to select Groq; OpenAI is the
default. The matching `OPENAI_API_KEY` or `GROQ_API_KEY` is required only for real
LLM chat requests. `LLM_API_KEY` remains available as a shared fallback.

## Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `APP_ENV` | Runtime environment label | `development` |
| `DATABASE_URL` | Async PostgreSQL URL | Local `rag` database |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `QDRANT_URL` | Qdrant HTTP URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Optional Qdrant authentication | unset |
| `QDRANT_COLLECTION` | Vector collection | `documents` |
| `LLM_PROVIDER` | `openai` or `groq` | `openai` |
| `OPENAI_API_KEY` | OpenAI API credential | unset |
| `GROQ_API_KEY` | Groq API credential | unset |
| `LLM_API_KEY` | Optional fallback credential for the selected provider | unset |
| `LLM_BASE_URL` | Optional OpenAI-compatible API base URL | Provider default |
| `LLM_MODEL` | Chat model name | `gpt-4o-mini` |
| `EMBEDDING_MODEL_NAME` | Local Sentence Transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_DIMENSION` | Vector dimension; must match model output | `384` |
| `MAX_UPLOAD_SIZE_BYTES` | Upload cap | `10485760` |
| `CHAT_HISTORY_LIMIT` | Number of recent user/assistant turns retained | `12` |
| `CHAT_SESSION_TTL_SECONDS` | Redis session expiration | `86400` |
| `RETRIEVAL_TOP_K` | Maximum matching chunks | `5` |
| `AUTO_CREATE_TABLES` | Create ORM tables at startup | `true` |

Never commit `.env` or pass credentials in source control. In Replit, store
`OPENAI_API_KEY` and/or `GROQ_API_KEY` in Secrets, then select the active provider
with `LLM_PROVIDER`.

## API

All application routes use the `/api/v1` prefix.

### Upload a document

`POST /api/v1/documents/upload` accepts multipart form data. Supported files are
PDF and UTF-8 TXT. The default maximum size is 10 MiB.

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@report.pdf" \
  -F "chunking_strategy=recursive" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200"
```

`chunking_strategy` is `fixed` or `recursive`. Both splitters validate that overlap
is smaller than the chunk size. Success returns the new document UUID, status,
filename, strategy, and chunk count.

### List, read, and delete documents

```bash
curl "http://localhost:8000/api/v1/documents?limit=50&offset=0"
curl http://localhost:8000/api/v1/documents/DOCUMENT_UUID
curl -X DELETE http://localhost:8000/api/v1/documents/DOCUMENT_UUID
```

Deletion removes matching Qdrant points and PostgreSQL metadata. A missing document
returns `404`; a storage failure returns an error so the delete can be retried.

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "candidate-session-123",
    "message": "Explain the main findings in the uploaded document"
  }'
```

The response includes `answer`, `session_id`, and source references with document ID,
filename, chunk index, and similarity score. Redis history is isolated by session ID
and limited by `CHAT_HISTORY_LIMIT`.

### Interview booking conversation

1. The user asks to book an interview.
2. The assistant gathers any missing name, email, date, and time.
3. Once complete, it displays the normalized details and asks for explicit
   confirmation. No booking has been saved yet.
4. Only a clear confirmation causes PostgreSQL persistence. A success response is
   returned only after the database write succeeds.
5. If persistence fails, the API returns an error and does not claim success.

The tool schema is `book_interview(name, email, date, time)`. Dates use
`YYYY-MM-DD`; times use 24-hour `HH:MM`. Email, date, and time are validated before
the pending details are presented.

## Storage

- **PostgreSQL:** document metadata and confirmed bookings.
- **Qdrant:** embedded text chunks with document ID, filename, and chunk index.
- **Redis:** recent chat turns and temporary booking confirmation state. Conversations
  are not stored in PostgreSQL.

Document IDs in vector payloads allow retrieval sources and document deletion to stay
associated with the metadata record.

## Tests

Tests use mock embedding, vector, LLM, and memory clients; they do not call paid APIs
or need running PostgreSQL, Redis, or Qdrant services.

```bash
cd backend
pip install -r requirements.txt
pytest -q
```

The suite covers PDF/TXT extraction, invalid uploads, both chunking strategies,
embedding and Qdrant adapter behavior, ingestion cleanup on partial failure,
PostgreSQL repository behavior with in-memory SQLite, Redis conversation memory,
multi-turn grounded answers, sources, booking detail collection, confirmation,
idempotency support, and invalid booking data.
