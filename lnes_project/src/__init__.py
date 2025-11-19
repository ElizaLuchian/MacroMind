"""Core modules for the Latent News Event Simulation project."""

from .agents import (
    BaseAgent,
    ContrarianAgent,
    FinBERTAgent,
    GroqAgent,
    MomentumAgent,
    NewsReactiveAgent,
    RandomAgent,
)
from .clusterer import ClusterResult, cluster_embeddings, plot_cluster_distribution
from .data_loader import load_news, load_prices, merge_news_and_prices
from .embedder import NewsEmbedder
from .metrics import agent_profitability, compute_directional_accuracy, summarize_metrics
from .preprocess import preprocess_news
from .simulator import SimulationResult, simulate_market

__all__ = [
    # Agents
    "BaseAgent",
    "RandomAgent",
    "MomentumAgent",
    "ContrarianAgent",
    "NewsReactiveAgent",
    "FinBERTAgent",
    "GroqAgent",
    # Data
    "load_news",
    "load_prices",
    "merge_news_and_prices",
    "preprocess_news",
    # Embedding & Clustering
    "NewsEmbedder",
    "cluster_embeddings",
    "ClusterResult",
    "plot_cluster_distribution",
    # Simulation
    "simulate_market",
    "SimulationResult",
    # Metrics
    "compute_directional_accuracy",
    "agent_profitability",
    "summarize_metrics",
]
