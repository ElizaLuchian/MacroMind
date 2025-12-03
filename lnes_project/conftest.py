"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to Python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# Session-level fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory."""
    return ROOT


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Return data directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Return test data directory."""
    test_dir = project_root / "tests" / "fixtures"
    test_dir.mkdir(exist_ok=True, parents=True)
    return test_dir


# =============================================================================
# Test data fixtures
# =============================================================================

@pytest.fixture
def sample_news_data() -> pd.DataFrame:
    """Generate sample news data for testing."""
    from tests.test_utils import generate_mock_news_data
    return generate_mock_news_data(n_rows=50, seed=42)


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Generate sample price data for testing."""
    from tests.test_utils import generate_mock_price_data
    return generate_mock_price_data(n_rows=50, seed=42)


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Generate sample embeddings for testing."""
    from tests.test_utils import generate_mock_embeddings
    return generate_mock_embeddings(n_samples=50, n_features=384, seed=42)


@pytest.fixture
def sample_cluster_labels() -> np.ndarray:
    """Generate sample cluster labels for testing."""
    from tests.test_utils import generate_mock_cluster_labels
    return generate_mock_cluster_labels(n_samples=50, n_clusters=3, seed=42)


@pytest.fixture
def sample_action_log() -> dict[str, list[str]]:
    """Generate sample action log for testing."""
    from tests.test_utils import generate_mock_action_log
    agents = ["Momentum", "Contrarian", "Random", "NewsReactive"]
    return generate_mock_action_log(agents=agents, n_timesteps=50, seed=42)


@pytest.fixture
def sample_prices_series() -> list[float]:
    """Generate sample price series for metrics testing."""
    np.random.seed(42)
    prices = [100.0]
    for _ in range(99):
        change = np.random.normal(0, 2)
        prices.append(max(prices[-1] + change, 1.0))
    return prices


# =============================================================================
# Agent fixtures
# =============================================================================

@pytest.fixture
def mock_random_agent():
    """Create mock Random agent."""
    from tests.test_utils import MockAgent
    return MockAgent(name="Random", strategy="random", seed=42)


@pytest.fixture
def mock_momentum_agent():
    """Create mock agent that always buys."""
    from tests.test_utils import MockAgent
    return MockAgent(name="Momentum", strategy="buy", seed=42)


@pytest.fixture
def mock_contrarian_agent():
    """Create mock agent that always sells."""
    from tests.test_utils import MockAgent
    return MockAgent(name="Contrarian", strategy="sell", seed=42)


# =============================================================================
# Mock model fixtures
# =============================================================================

@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    from tests.test_utils import MockNewsEmbedder
    return MockNewsEmbedder(backend="tfidf", embedding_dim=384, seed=42)


# =============================================================================
# Configuration fixtures
# =============================================================================

@pytest.fixture
def default_config() -> dict:
    """Return default experiment configuration."""
    return {
        "experiment": {
            "name": "test_experiment",
            "seed": 42,
        },
        "dataset": {
            "source": "small",
        },
        "embedder": {
            "backend": "tfidf",
            "embedding_dim": 384,
        },
        "clustering": {
            "k": 3,
            "method": "kmeans",
        },
        "agents": {
            "types": ["Random", "Momentum", "Contrarian"],
        },
        "simulator": {
            "alpha": 0.01,
            "noise_std": 0.0,
        },
    }


# =============================================================================
# Pytest hooks
# =============================================================================

def pytest_configure(config):
    """Configure pytest."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: mark test as slow to run"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring API keys"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring model download"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Skip slow tests by default unless --runslow is passed
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runintegration", action="store_true", default=False, help="run integration tests"
    )


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test."""
    np.random.seed(42)
    import random
    random.seed(42)


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for tests."""
    return tmp_path


# =============================================================================
# Mock external dependencies
# =============================================================================

@pytest.fixture
def mock_groq_api(monkeypatch):
    """Mock Groq API to avoid actual API calls."""
    class MockGroqResponse:
        def __init__(self, decision: str = "hold"):
            self.decision = decision
        
        @property
        def choices(self):
            class Choice:
                class Message:
                    def __init__(self, content):
                        self.content = content
                
                def __init__(self, content):
                    self.message = self.Message(content)
            
            return [Choice(self.decision)]
    
    class MockGroq:
        def __init__(self, api_key: str):
            self.api_key = api_key
        
        def chat_completions_create(self, **kwargs):
            return MockGroqResponse("hold")
    
    return MockGroq


@pytest.fixture
def mock_finbert(monkeypatch):
    """Mock FinBERT to avoid loading actual model."""
    class MockFinBERTPipeline:
        def __init__(self):
            pass
        
        def __call__(self, text: str):
            return [{
                "label": "positive",
                "score": 0.75,
            }]
    
    return MockFinBERTPipeline()

