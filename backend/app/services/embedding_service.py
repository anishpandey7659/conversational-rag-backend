import asyncio
import threading
from collections.abc import Callable, Sequence
from typing import Any

from app.core.exceptions import IntegrationUnavailableError
from app.core.settings import Settings


class EmbeddingService:
    """Create vectors with a local Sentence Transformers model."""

    def __init__(
        self,
        settings: Settings,
        model: Any | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.settings = settings
        self._model = model
        self._model_factory = model_factory
        self._load_lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                try:
                    if self._model_factory:
                        self._model = self._model_factory(
                            self.settings.embedding_model_name
                        )
                    else:
                        from sentence_transformers import SentenceTransformer

                        self._model = SentenceTransformer(
                            self.settings.embedding_model_name
                        )
                except Exception as exc:
                    raise IntegrationUnavailableError(
                        "The embedding model could not be loaded. Check model access "
                        "and EMBEDDING_MODEL_NAME."
                    ) from exc
        return self._model

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        try:
            vectors = await asyncio.to_thread(
                model.encode,
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            result = vectors.tolist()
            if any(len(vector) != self.settings.embedding_dimension for vector in result):
                raise ValueError(
                    "The embedding model output dimension does not match "
                    "EMBEDDING_DIMENSION."
                )
            return [[float(value) for value in vector] for vector in result]
        except IntegrationUnavailableError:
            raise
        except Exception as exc:
            raise IntegrationUnavailableError("Text embedding generation failed.") from exc

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]
