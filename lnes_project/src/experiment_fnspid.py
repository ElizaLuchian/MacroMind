"""Experiment runner for the FNSPID dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

from .agents import ContrarianAgent, MomentumAgent, NewsReactiveAgent, RandomAgent
from .clusterer import cluster_embeddings
from .data_loader import load_fnspid, merge_news_and_prices
from .embedder import NewsEmbedder
from .metrics import summarize_metrics
from .preprocess import preprocess_news
from .simulator import simulate_market

logger = logging.getLogger(__name__)


def run_fnspid_experiment(
    *,
    tickers: Sequence[str],
    start_date: str | None = None,
    end_date: str | None = None,
    news_limit: int | None = 5000,
    price_limit: int | None = 20000,
    embedder_kwargs: Dict[str, object] | None = None,
    dataset_name: str = "Zihan1004/FNSPID",
    cache_dir: Path | str | None = None,
    auth_token: str | None = None,
    local_news_csv: Path | str | None = None,
    local_price_csv: Path | str | None = None,
) -> Dict[str, object]:
    """Run the full pipeline on a subset of the FNSPID dataset.

    The current simulator expects a single continuous price series, therefore
    callers must provide a single ticker symbol (additional tickers can be
    benchmarked via separate calls).
    """

    if not tickers:
        raise ValueError("At least one ticker symbol is required for the FNSPID experiment.")

    news_df, price_df = load_fnspid(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        news_limit=news_limit,
        price_limit=price_limit,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        auth_token=auth_token,
        local_news_csv=local_news_csv,
        local_price_csv=local_price_csv,
    )

    unique_tickers = sorted(set(price_df["ticker"].astype(str).str.upper()))
    if len(unique_tickers) != 1:
        raise ValueError(
            "FNSPID experiment currently requires exactly one ticker after filtering. "
            f"Found: {unique_tickers}. Try narrowing the `tickers`, `start_date`, or `end_date` parameters."
        )

    processed_news = preprocess_news(news_df)
    merged = merge_news_and_prices(processed_news, price_df, join_on=("date", "ticker"))

    embedder = NewsEmbedder(**(embedder_kwargs or {}))
    embeddings = embedder.encode_news(merged["combined_text"])
    cluster_result = cluster_embeddings(embeddings, k=5)
    merged["cluster_id"] = cluster_result.labels

    sentiment_col = "sentiment_hint"
    if sentiment_col not in merged.columns:
        raise ValueError("Expected sentiment annotations to be present in the merged dataframe.")
    cluster_sentiment = (
        merged.groupby("cluster_id")[sentiment_col]
        .agg(lambda col: col.eq("positive").mean() - col.eq("negative").mean())
        .to_dict()
    )

    agents = [
        RandomAgent(name="random"),
        MomentumAgent(name="momentum"),
        ContrarianAgent(name="contrarian"),
        NewsReactiveAgent(name="news", cluster_sentiment=cluster_sentiment),
    ]

    sim_result = simulate_market(merged, agents, alpha=0.02, noise_std=0.001)
    metrics = summarize_metrics(
        reference_prices=merged["close"].tolist(),
        simulated_prices=sim_result.prices,
        action_log=sim_result.action_log,
        clusters=merged["cluster_id"].tolist(),
    )

    return {"simulation": sim_result, "metrics": metrics, "clusters": cluster_result, "merged": merged}

