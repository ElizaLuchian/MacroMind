"""Data loading utilities for the Latent News Event Simulation project."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

PathLike = Union[str, Path]


def _resolve_path(path: PathLike) -> Path:
    """Resolve a filesystem path to an absolute Path object."""
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved


def load_news(path: PathLike, limit: Optional[int] = None) -> pd.DataFrame:
    """Load the curated news dataset.

    Args:
        path: Path to a CSV containing at least ``date`` and ``headline`` columns.
        limit: Optional cap on the number of rows to keep (useful for sampling).

    Returns:
        Pandas DataFrame sorted by date with parsed ``date`` column.
    """
    file_path = _resolve_path(path)
    df = pd.read_csv(file_path)
    if "date" not in df.columns:
        raise ValueError("News data must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if limit:
        df = df.head(limit)
    return df


def load_prices(path: PathLike, limit: Optional[int] = None) -> pd.DataFrame:
    """Load the reference price series."""
    file_path = _resolve_path(path)
    df = pd.read_csv(file_path)
    if "date" not in df.columns:
        raise ValueError("Price data must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    numeric_cols = [col for col in ("open", "high", "low", "close", "volume") if col in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if limit:
        df = df.head(limit)
    return df


def merge_news_and_prices(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    how: str = "inner",
    join_on: Union[str, Sequence[str]] = "date",
) -> pd.DataFrame:
    """Align news and price data on the date column.

    Args:
        news_df: Cleaned news DataFrame.
        price_df: Price DataFrame.
        how: Merge strategy (``inner`` by default to enforce overlap).
        join_on: Column name(s) to align on. Defaults to ``"date"``.

    Returns:
        Combined DataFrame with suffixed column names for overlaps.
    """
    if news_df.empty or price_df.empty:
        raise ValueError("News and price DataFrames cannot be empty.")

    join_cols = [join_on] if isinstance(join_on, str) else list(join_on)

    news = news_df.copy()
    prices = price_df.copy()
    for column in join_cols:
        if column not in news.columns or column not in prices.columns:
            raise ValueError(f"Join column '{column}' is missing from one of the datasets.")
        news[column] = pd.to_datetime(news[column]) if column == "date" else news[column]
        prices[column] = pd.to_datetime(prices[column]) if column == "date" else prices[column]

    merged = pd.merge(news, prices, on=join_cols, how=how, suffixes=("_news", "_price"))
    if merged.empty:
        raise ValueError("Merged dataset is empty. Check the overlapping date range.")
    return merged.reset_index(drop=True)


_FNSPID_NEWS_PRIORITY: Mapping[str, Sequence[str]] = {
    "date": ("date", "Date", "publish_date", "PublishDate", "timestamp", "Datetime"),
    "headline": ("headline", "Headline", "title", "Title", "Article_title", "Article Title"),
    "body": ("body", "Body", "content", "Content", "Article", "article_text"),
    "ticker": ("ticker", "Ticker", "symbol", "Symbol", "Stock_symbol", "Stock Symbol"),
    "sentiment_hint": ("sentiment_hint", "sentiment_label", "Sentiment", "Sentiment_label", "Finbert_sentiment"),
}

_FNSPID_PRICE_PRIORITY: Mapping[str, Sequence[str]] = {
    "date": ("date", "Date", "timestamp"),
    "ticker": ("ticker", "Ticker", "symbol", "Symbol", "Stock_symbol", "Stock Symbol"),
    "open": ("open", "Open"),
    "high": ("high", "High"),
    "low": ("low", "Low"),
    "close": ("close", "Close", "AdjClose", "Adj Close"),
    "volume": ("volume", "Volume"),
}


def _rename_by_priority(df: pd.DataFrame, mapping: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    renamed = df.copy()
    for target, candidates in mapping.items():
        if target in renamed.columns:
            continue
        for candidate in candidates:
            if candidate in renamed.columns:
                renamed = renamed.rename(columns={candidate: target})
                break
    return renamed


def _ensure_datetime(value: Optional[Union[str, datetime, pd.Timestamp]]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    return pd.to_datetime(value)


def _normalize_sentiment(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    normalized = series.astype(str).str.lower().str.strip()
    mapping = {
        "pos": "positive",
        "positive": "positive",
        "neg": "negative",
        "negative": "negative",
        "neu": "neutral",
        "neutral": "neutral",
    }
    return normalized.map(lambda val: mapping.get(val, val))


def _slice_expression(split_name: str, limit: Optional[int]) -> str:
    if limit is None or limit <= 0:
        return split_name
    return f"{split_name}[:{limit}]"


def _load_fnspid_split(
    *,
    dataset_name: str,
    split: str,
    cache_dir: Optional[PathLike],
    auth_token: Optional[str],
    use_local_files_only: bool,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - guarded import
        raise ImportError(
            "The 'datasets' package is required to load FNSPID data. Install it via `pip install datasets`."
        ) from exc

    return load_dataset(
        dataset_name,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
        use_auth_token=auth_token,
        use_local_files_only=use_local_files_only,
    )


def load_fnspid(
    *,
    tickers: Optional[Sequence[str]] = None,
    start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    news_limit: Optional[int] = 5000,
    price_limit: Optional[int] = 20000,
    dataset_name: str = "Zihan1004/FNSPID",
    cache_dir: Optional[PathLike] = None,
    auth_token: Optional[str] = None,
    local_news_csv: Optional[PathLike] = None,
    local_price_csv: Optional[PathLike] = None,
    use_local_files_only: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load (a subset of) the FNSPID dataset via Hugging Face or local CSVs.

    Args:
        tickers: Optional collection of ticker symbols to keep (case-insensitive).
        start_date: Inclusive lower bound date filter (any pandas-compatible format).
        end_date: Inclusive upper bound date filter.
        news_limit: Number of news rows to request from Hugging Face (``None`` = full split).
        price_limit: Number of price rows to request from Hugging Face (``None`` = full split).
        dataset_name: Hugging Face dataset identifier (defaults to ``Zihan1004/FNSPID``).
        cache_dir: Optional Hugging Face cache directory.
        auth_token: Optional token for private datasets.
        local_news_csv: Optional path to a local CSV copy of the news table.
        local_price_csv: Optional path to a local CSV copy of the price table.
        use_local_files_only: Forwarded to ``datasets.load_dataset`` to force offline reads.

    Returns:
        A tuple of (news_df, price_df) already filtered, renamed, and date-parsed.
    """

    if local_news_csv:
        news_df = pd.read_csv(_resolve_path(local_news_csv))
    else:
        news_split = _slice_expression("news", news_limit)
        news_ds = _load_fnspid_split(
            dataset_name=dataset_name,
            split=news_split,
            cache_dir=cache_dir,
            auth_token=auth_token,
            use_local_files_only=use_local_files_only,
        )
        news_df = news_ds.to_pandas()

    if local_price_csv:
        price_df = pd.read_csv(_resolve_path(local_price_csv))
    else:
        price_split = _slice_expression("price", price_limit)
        price_ds = _load_fnspid_split(
            dataset_name=dataset_name,
            split=price_split,
            cache_dir=cache_dir,
            auth_token=auth_token,
            use_local_files_only=use_local_files_only,
        )
        price_df = price_ds.to_pandas()

    news_df = _rename_by_priority(news_df, _FNSPID_NEWS_PRIORITY)
    price_df = _rename_by_priority(price_df, _FNSPID_PRICE_PRIORITY)

    required_news_cols = {"date", "headline", "ticker"}
    missing_news = required_news_cols.difference(news_df.columns)
    if missing_news:
        raise ValueError(f"FNSPID news data is missing required columns: {missing_news}")

    required_price_cols = {"date", "ticker", "close"}
    missing_price = required_price_cols.difference(price_df.columns)
    if missing_price:
        raise ValueError(f"FNSPID price data is missing required columns: {missing_price}")

    news_df["date"] = pd.to_datetime(news_df["date"])
    price_df["date"] = pd.to_datetime(price_df["date"])
    if "body" not in news_df.columns:
        news_df["body"] = ""

    if "sentiment_hint" in news_df.columns:
        news_df["sentiment_hint"] = _normalize_sentiment(news_df["sentiment_hint"])
    else:
        score_col = next((col for col in news_df.columns if "sentiment" in col.lower()), None)
        if score_col:
            news_df["sentiment_hint"] = _normalize_sentiment(news_df[score_col])
        else:
            news_df["sentiment_hint"] = "neutral"

    ticker_set = {ticker.strip().upper() for ticker in tickers} if tickers else None
    if ticker_set:
        news_df = news_df[news_df["ticker"].astype(str).str.upper().isin(ticker_set)]
        price_df = price_df[price_df["ticker"].astype(str).str.upper().isin(ticker_set)]

    start_ts = _ensure_datetime(start_date)
    end_ts = _ensure_datetime(end_date)
    if start_ts is not None:
        news_df = news_df[news_df["date"] >= start_ts]
        price_df = price_df[price_df["date"] >= start_ts]
    if end_ts is not None:
        news_df = news_df[news_df["date"] <= end_ts]
        price_df = price_df[price_df["date"] <= end_ts]

    numeric_cols = [col for col in ("open", "high", "low", "close", "volume") if col in price_df.columns]
    price_df[numeric_cols] = price_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    news_df = news_df.sort_values("date").reset_index(drop=True)
    price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if news_df.empty:
        raise ValueError("FNSPID news selection is empty after filtering. Relax filters or increase limits.")
    if price_df.empty:
        raise ValueError("FNSPID price selection is empty after filtering. Relax filters or increase limits.")

    return news_df, price_df

