from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import Response

from app.core.dependencies import get_ingestion_service
from app.core.exceptions import AppError, NotFoundError
from app.core.settings import get_settings
from app.schemas.document import (
    ChunkingStrategy,
    DocumentListRead,
    DocumentRead,
    DocumentStatus,
    DocumentUploadRead,
)
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File()],
    chunking_strategy: Annotated[ChunkingStrategy, Form()] = ChunkingStrategy.recursive,
    chunk_size: Annotated[int, Form(ge=1, le=10_000)] = 1000,
    chunk_overlap: Annotated[int, Form(ge=0, le=9_999)] = 200,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentUploadRead:
    if chunk_overlap >= chunk_size:
        raise AppError(
            422,
            "invalid_chunking_options",
            "chunk_overlap must be smaller than chunk_size.",
        )
    settings = get_settings()
    content = await file.read(settings.max_upload_size_bytes + 1)
    document = await service.ingest(
        filename=file.filename,
        content_type=file.content_type,
        content=content,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return DocumentUploadRead(
        status=DocumentStatus(document.status),
        document_id=document.id,
        filename=document.filename,
        chunking_strategy=ChunkingStrategy(document.chunking_strategy),
        chunk_count=document.total_chunks,
    )


@router.get("", response_model=DocumentListRead)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentListRead:
    if not 1 <= limit <= 200 or offset < 0:
        raise AppError(422, "invalid_pagination", "Use limit 1–200 and offset 0 or greater.")
    items, total = await service.list(limit=limit, offset=offset)
    return DocumentListRead(
        items=[DocumentRead.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: UUID,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentRead:
    document = await service.get(document_id)
    return DocumentRead.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    service: IngestionService = Depends(get_ingestion_service),
) -> Response:
    await service.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
