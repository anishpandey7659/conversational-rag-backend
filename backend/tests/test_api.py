from fastapi.testclient import TestClient

from app.core.dependencies import get_ingestion_service
from app.main import app
from app.services.extraction_service import validate_upload


class UploadValidationService:
    async def ingest(
        self,
        filename: str | None,
        content_type: str | None,
        content: bytes,
        **_: object,
    ):
        validate_upload(filename, content, content_type, 1000)


def test_health_route_and_openapi_contract() -> None:
    client = TestClient(app)
    assert client.get("/healthz").json() == {"status": "ok"}
    schema = app.openapi()
    assert "/api/v1/documents/upload" in schema["paths"]
    assert "/api/v1/chat" in schema["paths"]


def test_document_upload_route_returns_error_for_unsupported_type() -> None:
    app.dependency_overrides[get_ingestion_service] = lambda: UploadValidationService()
    try:
        response = TestClient(app).post(
            "/api/v1/documents/upload",
            files={"file": ("payload.exe", b"not supported", "application/octet-stream")},
        )
    finally:
        app.dependency_overrides.pop(get_ingestion_service, None)

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "invalid_upload"


def test_chat_rejects_blank_messages() -> None:
    response = TestClient(app).post(
        "/api/v1/chat",
        json={"session_id": "session", "message": "   "},
    )
    assert response.status_code == 422
