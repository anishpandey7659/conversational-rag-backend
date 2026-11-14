import pymupdf
import pytest

from app.core.exceptions import InvalidUploadError
from app.schemas.document import ChunkingStrategy
from app.services.chunking_service import (
    fixed_size_chunks,
    recursive_character_chunks,
    split_text,
)
from app.services.extraction_service import ExtractionService, validate_upload


def test_txt_extraction_and_signature_validation() -> None:
    payload = b"\xef\xbb\xbfHello from a text file.\nSecond line."
    assert validate_upload("notes.txt", payload, "text/plain", 1000) == "txt"
    assert ExtractionService.extract("notes.txt", payload) == (
        "Hello from a text file.\nSecond line."
    )


def test_pdf_text_extraction() -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "PDF extraction test.")
    payload = pdf.tobytes()
    pdf.close()

    assert validate_upload("sample.pdf", payload, "application/pdf", 100_000) == "pdf"
    assert "PDF extraction test." in ExtractionService.extract("sample.pdf", payload)


@pytest.mark.parametrize(
    ("filename", "payload", "content_type"),
    [
        ("script.exe", b"MZ", "application/octet-stream"),
        ("broken.pdf", b"not a pdf", "application/pdf"),
        ("empty.txt", b"", "text/plain"),
        ("notes.txt", b"\xff", "text/plain"),
    ],
)
def test_invalid_uploads_are_rejected(
    filename: str, payload: bytes, content_type: str
) -> None:
    with pytest.raises(InvalidUploadError):
        if filename.endswith(".txt") and payload:
            ExtractionService.extract(filename, payload)
        else:
            validate_upload(filename, payload, content_type, 100)


def test_upload_size_limit_is_enforced() -> None:
    with pytest.raises(InvalidUploadError, match="exceeds"):
        validate_upload("large.txt", b"12345", "text/plain", 4)


def test_fixed_chunking_respects_size_and_overlap() -> None:
    chunks = fixed_size_chunks("abcdefghij", chunk_size=5, chunk_overlap=2)
    assert chunks == ["abcde", "defgh", "ghij"]
    assert all(len(chunk) <= 5 for chunk in chunks)


def test_recursive_chunking_prefers_word_boundaries() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = recursive_character_chunks(text, chunk_size=16, chunk_overlap=3)
    assert len(chunks) > 1
    assert all(len(chunk) <= 16 for chunk in chunks)
    assert split_text(text, ChunkingStrategy.recursive, 16, 3) == chunks
    assert all(chunk.strip() == chunk for chunk in chunks)


def test_chunking_options_are_validated() -> None:
    with pytest.raises(ValueError, match="smaller"):
        fixed_size_chunks("some text", 4, 4)
