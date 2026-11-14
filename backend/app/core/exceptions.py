class AppError(Exception):
    """An expected application error safe to expose through the API."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class InvalidUploadError(AppError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(status_code, "invalid_upload", message)


class NotFoundError(AppError):
    def __init__(self, resource: str) -> None:
        super().__init__(404, "not_found", f"{resource} was not found.")


class IntegrationUnavailableError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(503, "integration_unavailable", message)


class DataIntegrityError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(409, "data_integrity_error", message)
