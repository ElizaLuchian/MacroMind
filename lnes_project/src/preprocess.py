"""News preprocessing helpers."""

from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd

_NON_ALPHA_NUMERIC = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize textual inputs."""
    if not isinstance(text, str):
        return ""
    normalized = text.lower()
    normalized = _NON_ALPHA_NUMERIC.sub(" ", normalized)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    return normalized


def preprocess_news(
    df: pd.DataFrame,
    text_columns: Iterable[str] = ("headline", "body"),
) -> pd.DataFrame:
    """Apply cleaning to the relevant textual columns and aggregate them."""
    if df.empty:
        raise ValueError("Cannot preprocess an empty DataFrame.")

    processed = df.copy()
    clean_columns: List[str] = []
    for column in text_columns:
        if column not in processed.columns:
            continue
        clean_column = f"{column}_clean"
        processed[clean_column] = processed[column].fillna("").map(clean_text)
        clean_columns.append(clean_column)

    if not clean_columns:
        raise ValueError("No valid text columns were found for preprocessing.")

    processed["combined_text"] = (
        processed[clean_columns].apply(lambda row: " ".join(filter(None, row)), axis=1).str.strip()
    )
    return processed




