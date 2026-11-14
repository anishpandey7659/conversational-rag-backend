from contextlib import asynccontextmanager
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1.chat import router as chat_router
from app.api.v1.documents import router as documents_router
from app.api.v1.health import router as health_router
from app.core.dependencies import get_llm_client, get_qdrant_client, get_redis_client
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.db.base import Base
from app.db.session import get_engine
from app import models  # noqa: F401 - register ORM mappings with Base.metadata

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.auto_create_tables:
        async with get_engine().begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    try:
        yield
    finally:
        if get_llm_client.cache_info().currsize:
            try:
                await get_llm_client().close()
            except Exception:
                logger.debug("LLM client was not initialized cleanly during shutdown")
        if get_qdrant_client.cache_info().currsize:
            try:
                await get_qdrant_client().close()
            except Exception:
                logger.debug("Qdrant client was not initialized cleanly during shutdown")
        if get_redis_client.cache_info().currsize:
            try:
                await get_redis_client().aclose()
            except Exception:
                logger.debug("Redis client was not initialized cleanly during shutdown")
        await get_engine().dispose()


app = FastAPI(
    title="Conversational RAG Backend",
    description="Document ingestion, grounded conversational retrieval, and interview booking.",
    version="1.0.0",
    lifespan=lifespan,
)

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(documents_router)
v1_router.include_router(chat_router)
app.include_router(v1_router)
app.include_router(health_router)


@app.exception_handler(AppError)
async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )
