"""Evaluation metrics for the latent news simulation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ActionLog = Mapping[str, Sequence[str]]

ACTION_TO_SIGN = {"buy": 1, "sell": -1, "hold": 0}


def compute_directional_accuracy(reference: Sequence[float], simulated: Sequence[float]) -> float:
    """Share of days where simulated direction matches reference direction."""
    if len(reference) != len(simulated):
        raise ValueError("Reference and simulated series must have the same length.")
    if len(reference) < 2:
        return 0.0
    ref_diff = np.diff(reference)
    sim_diff = np.diff(simulated)
    matches = np.sign(ref_diff) == np.sign(sim_diff)
    return float(matches.sum() / len(matches))


def agent_profitability(action_log: ActionLog, price_series: Sequence[float]) -> pd.DataFrame:
    """Compute a naive profitability metric per agent."""
    if len(price_series) < 2:
        raise ValueError("Price series must contain at least two values.")
    returns = np.diff(price_series)
    results: List[Dict[str, float]] = []
    for name, actions in action_log.items():
        signals = np.array([ACTION_TO_SIGN.get(a, 0) for a in actions[: len(returns)]])
        pnl = float(np.dot(signals, returns))
        accuracy = float((signals == np.sign(returns)).mean())
        results.append({"agent": name, "pnl": pnl, "directional_accuracy": accuracy})
    return pd.DataFrame(results)


def volatility_clustering(prices: Sequence[float], window: int = 5) -> float:
    """Estimate autocorrelation of squared returns as a proxy for volatility clustering."""
    returns = np.diff(prices)
    if len(returns) <= window:
        return 0.0
    squared = returns**2
    series = pd.Series(squared)
    return float(series.autocorr(lag=window))


def cluster_price_correlation(clusters: Sequence[int], prices: Sequence[float]) -> float:
    """Compute Pearson correlation between clusters and price changes."""
    if len(clusters) != len(prices):
        raise ValueError("Clusters and prices must have the same length.")
    if len(prices) < 2:
        return 0.0
    price_changes = np.diff(prices)
    clipped_clusters = np.array(clusters[1:], dtype=float)
    corr, _ = pearsonr(clipped_clusters, price_changes)
    return float(corr)


def decision_correlation_matrix(action_log: ActionLog) -> pd.DataFrame:
    """Correlation of discrete agent decisions."""
    encoded = {agent: [ACTION_TO_SIGN.get(action, 0) for action in actions] for agent, actions in action_log.items()}
    df = pd.DataFrame(encoded)
    if df.empty:
        raise ValueError("Action log is empty.")
    return df.corr()


def summarize_metrics(
    reference_prices: Sequence[float],
    simulated_prices: Sequence[float],
    action_log: ActionLog,
    clusters: Sequence[int],
) -> Dict[str, object]:
    """Bundle the key summary statistics."""
    summary = {
        "directional_accuracy": compute_directional_accuracy(reference_prices, simulated_prices),
        "volatility_clustering": volatility_clustering(simulated_prices),
        "cluster_price_correlation": cluster_price_correlation(clusters, simulated_prices),
        "agent_profitability": agent_profitability(action_log, simulated_prices),
        "decision_correlation": decision_correlation_matrix(action_log),
    }
    return summary




