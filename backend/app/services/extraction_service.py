import pymupdf

from app.core.exceptions import InvalidUploadError


class ExtractionService:
    """Extract plain text from supported upload formats."""

    @staticmethod
    def extract(filename: str, content: bytes) -> str:
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        try:
            if extension == "pdf":
                with pymupdf.open(stream=content, filetype="pdf") as document:
                    text = "\n".join(page.get_text("text") for page in document)
            elif extension == "txt":
                text = content.decode("utf-8-sig")
            else:
                raise InvalidUploadError("Only PDF and TXT files are supported.", 415)
        except InvalidUploadError:
            raise
        except (
            UnicodeDecodeError,
            pymupdf.FileDataError,
            pymupdf.EmptyFileError,
        ) as exc:
            raise InvalidUploadError(
                "The file could not be read. Check that it is a valid UTF-8 TXT or PDF."
            ) from exc
        except Exception as exc:
            raise InvalidUploadError("Text extraction failed for this file.") from exc

        normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        if not normalized:
            raise InvalidUploadError("The document contains no extractable text.", 422)
        return normalized


def validate_upload(
    filename: str | None,
    content: bytes,
    content_type: str | None,
    max_size_bytes: int,
) -> str:
    """Validate upload name, format, size, and basic file signature."""
    if not filename or not filename.strip():
        raise InvalidUploadError("A filename is required.")
    if not content:
        raise InvalidUploadError("The uploaded file is empty.")
    if len(content) > max_size_bytes:
        raise InvalidUploadError(
            f"The file exceeds the {max_size_bytes}-byte upload limit.", 413
        )

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in {"pdf", "txt"}:
        raise InvalidUploadError("Only PDF and TXT files are supported.", 415)
    if content_type and content_type not in {
        "application/pdf",
        "text/plain",
        "application/octet-stream",
        "application/x-pdf",
    }:
        raise InvalidUploadError("The uploaded content type is not supported.", 415)
    if extension == "pdf" and not content.lstrip().startswith(b"%PDF-"):
        raise InvalidUploadError("The uploaded file does not have a valid PDF signature.")
    return extension
