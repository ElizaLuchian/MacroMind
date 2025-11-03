"""Embedding utilities leveraging Sentence Transformers with optional fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


SUPPORTED_MODELS = {
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "finbert": "ProsusAI/finbert",
}


def _resolve_model_name(name: Optional[str]) -> str:
    if not name:
        return SUPPORTED_MODELS["all-minilm"]
    key = name.lower()
    return SUPPORTED_MODELS.get(key, name)


@dataclass
class NewsEmbedder:
    """Wrapper around embedding backends with sensible defaults."""

    model_name: Optional[str] = None
    batch_size: int = 32
    normalize: bool = True
    backend: str = "sentence-transformer"
    random_state: Optional[int] = None
    _model: Optional[object] = field(default=None, init=False, repr=False)
    _vectorizer_fitted: bool = field(default=False, init=False, repr=False)

    def load(self) -> None:
        """Load the requested backend."""
        if self._model is not None:
            return

        if self.backend == "sentence-transformer":
            if SentenceTransformer is None:
                logger.warning(
                    "sentence-transformers is unavailable; "
                    "falling back to TF-IDF embeddings. Install sentence-transformers for better quality."
                )
                self.backend = "tfidf"
                return self.load()

            model_path = _resolve_model_name(self.model_name)
            logger.info("Loading SentenceTransformer model %s", model_path)
            self._model = SentenceTransformer(model_path)
            return

        if self.backend == "tfidf":
            rng_seed = self.random_state or 0
            self._model = TfidfVectorizer(max_features=384, ngram_range=(1, 2), stop_words="english")
            np.random.default_rng(rng_seed)  # ensure deterministic init for coverage
            return

        raise ValueError(f"Unsupported backend: {self.backend}")

    def encode_news(self, texts: Sequence[str] | Iterable[str]) -> np.ndarray:
        """Encode a list of texts into dense vectors."""
        if not isinstance(texts, Iterable):
            raise TypeError("Texts must be an iterable of strings.")
        text_list = list(texts)
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)
        if self._model is None:
            self.load()
        assert self._model is not None  # for mypy

        if self.backend == "sentence-transformer":
            embeddings = self._model.encode(
                text_list,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype=np.float32)

        if self.backend == "tfidf":
            model: TfidfVectorizer = self._model  # type: ignore[assignment]
            if not self._vectorizer_fitted:
                matrix = model.fit_transform(text_list)
                self._vectorizer_fitted = True
            else:
                matrix = model.transform(text_list)
            embeddings = matrix.toarray()
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
                embeddings = embeddings / norms
            return embeddings.astype(np.float32)

        raise RuntimeError("Embedding backend is not initialized correctly.")

