"""Test utilities: mock data generators, fixtures, and assertion helpers."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


def generate_mock_news_data(
    n_rows: int = 100,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate mock news data for testing.
    
    Args:
        n_rows: Number of news items to generate
        start_date: Starting date for news data
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: date, headline, body, ticker
    """
    random.seed(seed)
    np.random.seed(seed)
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    
    headlines = [
        "Company reports strong earnings",
        "Stock price reaches new high",
        "CEO announces new product line",
        "Analysts upgrade price target",
        "Market volatility concerns investors",
        "Regulatory approval received",
        "Merger talks confirmed",
        "Quarterly revenue beats expectations",
        "Supply chain disruptions reported",
        "Technology breakthrough announced",
    ]
    
    bodies = [
        "The company reported strong quarterly earnings today.",
        "Investors reacted positively to the news.",
        "Market analysts are optimistic about future growth.",
        "The stock showed significant movement today.",
        "Industry experts weigh in on the developments.",
        "Shareholders approved the proposal.",
        "Trading volume increased substantially.",
        "The announcement came after market close.",
        "Economic indicators suggest continued momentum.",
        "Management expressed confidence in the strategy.",
    ]
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    data = []
    for i in range(n_rows):
        date = start + timedelta(days=i % 100)
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "headline": random.choice(headlines),
            "body": random.choice(bodies),
            "ticker": random.choice(tickers),
        })
    
    return pd.DataFrame(data)


def generate_mock_price_data(
    n_rows: int = 100,
    start_date: str = "2023-01-01",
    start_price: float = 100.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate mock price data for testing.
    
    Args:
        n_rows: Number of days to generate
        start_date: Starting date
        start_price: Initial price
        volatility: Daily volatility (std dev of returns)
        seed: Random seed
        
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    np.random.seed(seed)
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Generate random walk prices
    returns = np.random.normal(0, volatility, n_rows)
    prices = start_price * np.cumprod(1 + returns)
    
    data = []
    for i in range(n_rows):
        date = start + timedelta(days=i)
        close = prices[i]
        # Generate OHLC data
        daily_range = close * np.random.uniform(0.01, 0.03)
        low = close - daily_range * np.random.uniform(0, 1)
        high = close + daily_range * np.random.uniform(0, 1)
        open_price = np.random.uniform(low, high)
        volume = int(np.random.uniform(1e6, 10e6))
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": volume,
        })
    
    return pd.DataFrame(data)


def generate_mock_embeddings(
    n_samples: int = 100,
    n_features: int = 384,
    seed: int = 42,
) -> np.ndarray:
    """Generate mock embedding vectors for testing.
    
    Args:
        n_samples: Number of embedding vectors
        n_features: Dimension of each embedding
        seed: Random seed
        
    Returns:
        Array of shape (n_samples, n_features)
    """
    np.random.seed(seed)
    # Generate embeddings with some structure (clusterable)
    n_clusters = 3
    cluster_centers = np.random.randn(n_clusters, n_features)
    
    embeddings = []
    for i in range(n_samples):
        cluster_id = i % n_clusters
        embedding = cluster_centers[cluster_id] + np.random.randn(n_features) * 0.1
        embeddings.append(embedding)
    
    return np.array(embeddings)


def generate_mock_cluster_labels(
    n_samples: int = 100,
    n_clusters: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Generate mock cluster labels for testing.
    
    Args:
        n_samples: Number of samples
        n_clusters: Number of clusters
        seed: Random seed
        
    Returns:
        Array of cluster labels
    """
    np.random.seed(seed)
    return np.random.randint(0, n_clusters, size=n_samples)


def generate_mock_action_log(
    agents: list[str],
    n_timesteps: int = 100,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate mock agent action log for testing.
    
    Args:
        agents: List of agent names
        n_timesteps: Number of timesteps
        seed: Random seed
        
    Returns:
        Dictionary mapping agent names to lists of actions
    """
    random.seed(seed)
    actions = ["buy", "sell", "hold"]
    
    action_log = {}
    for agent in agents:
        # Each agent has different action probabilities
        if "momentum" in agent.lower():
            # Momentum tends to buy
            weights = [0.5, 0.2, 0.3]
        elif "contrarian" in agent.lower():
            # Contrarian tends to sell
            weights = [0.2, 0.5, 0.3]
        elif "random" in agent.lower():
            # Random is uniform
            weights = [0.33, 0.33, 0.34]
        else:
            weights = [0.4, 0.3, 0.3]
        
        action_log[agent] = random.choices(actions, weights=weights, k=n_timesteps)
    
    return action_log


def assert_valid_price_series(prices: pd.Series | list | np.ndarray) -> None:
    """Assert that a price series is valid.
    
    Args:
        prices: Price series to validate
        
    Raises:
        AssertionError: If price series is invalid
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    elif isinstance(prices, list):
        prices = np.array(prices)
    
    assert len(prices) > 0, "Price series is empty"
    assert np.all(prices >= 0), "Prices must be non-negative"
    assert not np.any(np.isnan(prices)), "Prices contain NaN values"
    assert not np.any(np.isinf(prices)), "Prices contain infinite values"


def assert_valid_embeddings(embeddings: np.ndarray) -> None:
    """Assert that embeddings are valid.
    
    Args:
        embeddings: Embedding matrix to validate
        
    Raises:
        AssertionError: If embeddings are invalid
    """
    assert embeddings.ndim == 2, f"Embeddings must be 2D, got {embeddings.ndim}D"
    assert embeddings.shape[0] > 0, "Embeddings matrix is empty"
    assert embeddings.shape[1] > 0, "Embeddings have zero features"
    assert not np.any(np.isnan(embeddings)), "Embeddings contain NaN values"
    assert not np.any(np.isinf(embeddings)), "Embeddings contain infinite values"


def assert_valid_cluster_labels(labels: np.ndarray, n_clusters: int | None = None) -> None:
    """Assert that cluster labels are valid.
    
    Args:
        labels: Cluster label array
        n_clusters: Expected number of clusters (optional)
        
    Raises:
        AssertionError: If labels are invalid
    """
    assert labels.ndim == 1, f"Labels must be 1D, got {labels.ndim}D"
    assert len(labels) > 0, "Labels array is empty"
    assert np.all(labels >= 0), "Labels must be non-negative"
    assert np.all(labels == labels.astype(int)), "Labels must be integers"
    
    if n_clusters is not None:
        unique_labels = set(labels)
        assert len(unique_labels) <= n_clusters, \
            f"Found {len(unique_labels)} unique labels, expected <= {n_clusters}"


def assert_valid_action_log(action_log: dict[str, list[str]]) -> None:
    """Assert that action log is valid.
    
    Args:
        action_log: Action log dictionary
        
    Raises:
        AssertionError: If action log is invalid
    """
    assert len(action_log) > 0, "Action log is empty"
    
    valid_actions = {"buy", "sell", "hold"}
    lengths = []
    
    for agent, actions in action_log.items():
        assert isinstance(agent, str), f"Agent name must be string, got {type(agent)}"
        assert len(actions) > 0, f"Actions for agent {agent} is empty"
        lengths.append(len(actions))
        
        for action in actions:
            assert action in valid_actions, \
                f"Invalid action '{action}' for agent {agent}, must be one of {valid_actions}"
    
    # All agents should have same number of actions
    assert len(set(lengths)) == 1, \
        f"Agents have different number of actions: {lengths}"


def assert_metrics_in_range(metrics: dict[str, float], expected_ranges: dict[str, tuple[float, float]]) -> None:
    """Assert that metrics are within expected ranges.
    
    Args:
        metrics: Dictionary of metric values
        expected_ranges: Dictionary of (min, max) tuples for each metric
        
    Raises:
        AssertionError: If any metric is out of range
    """
    for metric_name, (min_val, max_val) in expected_ranges.items():
        assert metric_name in metrics, f"Metric '{metric_name}' not found in results"
        value = metrics[metric_name]
        assert min_val <= value <= max_val, \
            f"Metric '{metric_name}' = {value} out of range [{min_val}, {max_val}]"


def create_mock_dataframe(
    columns: list[str],
    n_rows: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a mock DataFrame with specified columns.
    
    Args:
        columns: List of column names
        n_rows: Number of rows
        seed: Random seed
        
    Returns:
        DataFrame with random data
    """
    np.random.seed(seed)
    
    data = {}
    for col in columns:
        if "date" in col.lower():
            start = datetime(2023, 1, 1)
            data[col] = [start + timedelta(days=i) for i in range(n_rows)]
        elif "price" in col.lower() or "close" in col.lower():
            data[col] = np.random.uniform(50, 150, n_rows)
        elif "volume" in col.lower():
            data[col] = np.random.randint(1e6, 10e6, n_rows)
        elif "cluster" in col.lower() or "label" in col.lower():
            data[col] = np.random.randint(0, 3, n_rows)
        else:
            data[col] = np.random.randn(n_rows)
    
    return pd.DataFrame(data)


class MockNewsEmbedder:
    """Mock NewsEmbedder for testing without loading models."""
    
    def __init__(self, backend: str = "tfidf", embedding_dim: int = 384, seed: int = 42):
        self.backend = backend
        self.embedding_dim = embedding_dim
        self.seed = seed
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate mock embeddings."""
        return generate_mock_embeddings(
            n_samples=len(texts),
            n_features=self.embedding_dim,
            seed=self.seed,
        )


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name: str, strategy: str = "random", seed: int = 42):
        self.name = name
        self.strategy = strategy
        self.seed = seed
        random.seed(seed)
    
    def decide(self, market_state: dict[str, Any]) -> str:
        """Make mock decision."""
        if self.strategy == "random":
            return random.choice(["buy", "sell", "hold"])
        elif self.strategy == "buy":
            return "buy"
        elif self.strategy == "sell":
            return "sell"
        elif self.strategy == "hold":
            return "hold"
        else:
            return random.choice(["buy", "sell", "hold"])


def is_sorted(arr: list | np.ndarray, ascending: bool = True) -> bool:
    """Check if array is sorted.
    
    Args:
        arr: Array to check
        ascending: Whether to check for ascending order
        
    Returns:
        True if array is sorted
    """
    if len(arr) <= 1:
        return True
    
    if ascending:
        return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
    else:
        return all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))


def count_unique(arr: list | np.ndarray) -> int:
    """Count unique elements in array.
    
    Args:
        arr: Array to count
        
    Returns:
        Number of unique elements
    """
    return len(set(arr))


def random_string(length: int = 10, seed: int | None = None) -> str:
    """Generate random string for testing.
    
    Args:
        length: Length of string
        seed: Random seed
        
    Returns:
        Random string
    """
    if seed is not None:
        random.seed(seed)
    
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(chars) for _ in range(length))

