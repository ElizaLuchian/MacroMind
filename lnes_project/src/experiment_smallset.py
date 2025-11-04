"""Small dataset experiment orchestrating the full pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .agents import ContrarianAgent, MomentumAgent, NewsReactiveAgent, RandomAgent
from .clusterer import cluster_embeddings
from .data_loader import load_news, load_prices, merge_news_and_prices
from .embedder import NewsEmbedder
from .metrics import summarize_metrics
from .preprocess import preprocess_news
from .simulator import simulate_market

logger = logging.getLogger(__name__)


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def run_small_dataset_experiment(
    data_dir: Path | str | None = None,
    embedder_kwargs: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Execute the baseline experiment."""
    directory = Path(data_dir) if data_dir else _default_data_dir()
    news = preprocess_news(load_news(directory / "small_news.csv"))
    prices = load_prices(directory / "small_prices.csv")
    merged = merge_news_and_prices(news, prices)

    embedder = NewsEmbedder(**(embedder_kwargs or {}))
    embeddings = embedder.encode_news(merged["combined_text"])
    cluster_result = cluster_embeddings(embeddings, k=3)
    merged["cluster_id"] = cluster_result.labels

    cluster_sentiment = (
        merged.groupby("cluster_id")["sentiment_hint"].agg(lambda col: col.eq("positive").mean() - col.eq("negative").mean())
    )

    agents = [
        RandomAgent(name="random"),
        MomentumAgent(name="momentum"),
        ContrarianAgent(name="contrarian"),
        NewsReactiveAgent(name="news", cluster_sentiment=cluster_sentiment.to_dict()),
    ]

    sim_result = simulate_market(merged, agents, alpha=0.02, noise_std=0.001)
    metrics = summarize_metrics(
        reference_prices=merged["close"].tolist(),
        simulated_prices=sim_result.prices,
        action_log=sim_result.action_log,
        clusters=merged["cluster_id"].tolist(),
    )
    return {"simulation": sim_result, "metrics": metrics, "clusters": cluster_result, "merged": merged}

