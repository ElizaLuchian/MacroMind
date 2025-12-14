# Contributing to MacroMind

Thank you for your interest in contributing to the **Latent News Event Simulation (MacroMind)** project! This guide will help you extend the codebase with new agents, metrics, clustering algorithms, and datasets.

---

## Table of Contents

1. [Code Structure Overview](#code-structure-overview)
2. [Development Setup](#development-setup)
3. [How to Add New Agents](#how-to-add-new-agents)
4. [How to Add New Metrics](#how-to-add-new-metrics)
5. [How to Add New Clustering Algorithms](#how-to-add-new-clustering-algorithms)
6. [How to Integrate New Datasets](#how-to-integrate-new-datasets)
7. [Testing Requirements](#testing-requirements)
8. [Code Style Guidelines](#code-style-guidelines)
9. [Pull Request Process](#pull-request-process)
10. [Community Guidelines](#community-guidelines)

---

## Code Structure Overview

```
lnes_project/
├── src/                        # Core library modules
│   ├── data_loader.py          # Data ingestion (CSV, FNSPID, custom)
│   ├── preprocess.py           # Text cleaning and normalization
│   ├── embedder.py             # Sentence transformers + TF-IDF
│   ├── clusterer.py            # K-Means clustering + visualization
│   ├── agents.py               # Agent implementations (rule-based + AI)
│   ├── simulator.py            # Market simulation engine
│   ├── metrics.py              # Performance evaluation metrics
│   ├── experiment_smallset.py  # Small dataset experiment runner
│   └── experiment_fnspid.py    # FNSPID dataset experiment runner
├── tests/                      # Unit and integration tests
│   ├── test_agents.py          # Agent behavior tests
│   ├── test_clusterer.py       # Clustering tests
│   ├── test_data_loader.py     # Data loading tests
│   ├── test_embedder.py        # Embedding tests
│   ├── test_simulator.py       # Simulation tests
│   └── test_integration.py     # End-to-end pipeline tests
├── scripts/                    # Executable experiment scripts
│   ├── run_experiment.py       # Main CLI for running experiments
│   ├── run_ai_experiment.py    # AI agent specific experiments
│   └── ...                     # Other utility scripts
├── data/                       # Sample datasets
│   ├── small_news.csv          # ~100 rows curated news
│   ├── small_prices.csv        # ~100 rows curated prices
│   └── ...
├── docs/                       # Documentation
│   ├── experimental_design.md  # Formal methodology
│   ├── REPRODUCIBILITY.md      # Reproduction guide
│   └── ARCHITECTURE.md         # System design
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
```

### Module Dependencies

```
data_loader → preprocess → embedder → clusterer
                    ↓
                  agents → simulator → metrics
```

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/MacroMind.git
cd MacroMind/lnes_project
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# For development (adds pytest, linters)
pip install pytest pytest-cov black flake8 mypy
```

### 4. Verify Setup

```bash
# Run tests
pytest

# Run small experiment
python -c "from src.experiment_smallset import run_small_dataset_experiment; run_small_dataset_experiment()"
```

---

## How to Add New Agents

Agents make trading decisions based on market state and news information. All agents inherit from `BaseAgent`.

### Step 1: Define Your Agent Class

**File**: `src/agents.py`

```python
from dataclasses import dataclass
from src.agents import BaseAgent, Action, MarketState
from typing import Optional

@dataclass
class MyCustomAgent(BaseAgent):
    """Your agent description here."""
    
    # Add custom parameters
    custom_param: float = 0.5
    
    def decide(
        self, 
        market_state: MarketState, 
        cluster_id: Optional[int],
        news_text: Optional[str] = None
    ) -> Action:
        """
        Make a trading decision.
        
        Args:
            market_state: Dict with 'price', 'prev_price', 'volume'
            cluster_id: Latent event cluster (0, 1, 2, ...)
            news_text: Raw news text (for AI agents)
            
        Returns:
            One of: "buy", "sell", "hold"
        """
        # Your logic here
        price = market_state.get("price", 0.0)
        prev_price = market_state.get("prev_price", 0.0)
        
        # Example: Buy if price increases by >1%
        if price > prev_price * 1.01:
            return "buy"
        elif price < prev_price * 0.99:
            return "sell"
        return "hold"
```

### Step 2: Add Tests

**File**: `tests/test_agents.py`

```python
def test_my_custom_agent():
    """Test MyCustomAgent behavior."""
    agent = MyCustomAgent(name="TestAgent", custom_param=0.8, seed=42)
    
    # Test buy condition
    market_state = {"price": 101.0, "prev_price": 100.0}
    action = agent.decide(market_state, cluster_id=None)
    assert action == "buy"
    
    # Test sell condition
    market_state = {"price": 99.0, "prev_price": 100.0}
    action = agent.decide(market_state, cluster_id=None)
    assert action == "sell"
    
    # Test hold condition
    market_state = {"price": 100.5, "prev_price": 100.0}
    action = agent.decide(market_state, cluster_id=None)
    assert action == "hold"
```

### Step 3: Integrate into Experiments

**File**: `scripts/run_experiment.py` (or custom script)

```python
from src.agents import MyCustomAgent

agents = [
    MyCustomAgent(name="Custom_v1", custom_param=0.5, seed=42),
    MyCustomAgent(name="Custom_v2", custom_param=0.8, seed=42),
    MomentumAgent(name="Momentum_Baseline", seed=42),
]

result = simulate_market(merged_df, agents, alpha=0.01, seed=42)
```

### Agent Best Practices

1. **Reproducibility**: Always accept and use `seed` parameter
2. **Error Handling**: Return `"hold"` on invalid inputs
3. **Efficiency**: Keep `decide()` fast (<10ms for rule-based, <500ms for AI)
4. **Documentation**: Add docstrings explaining strategy
5. **Testing**: Cover edge cases (missing data, extreme prices)

### Example: Mean Reversion Agent

```python
@dataclass
class MeanReversionAgent(BaseAgent):
    """Trades based on deviation from moving average."""
    
    window: int = 20
    threshold: float = 0.05
    _price_history: List[float] = field(default_factory=list, init=False, repr=False)
    
    def decide(self, market_state: MarketState, cluster_id: Optional[int]) -> Action:
        price = market_state.get("price")
        if price is None:
            return "hold"
        
        self._price_history.append(price)
        if len(self._price_history) < self.window:
            return "hold"
        
        # Keep only recent window
        self._price_history = self._price_history[-self.window:]
        
        mean = sum(self._price_history) / len(self._price_history)
        deviation = (price - mean) / mean
        
        if deviation > self.threshold:
            return "sell"  # Price above mean, expect reversion
        elif deviation < -self.threshold:
            return "buy"   # Price below mean, expect reversion
        return "hold"
```

---

## How to Add New Metrics

Metrics quantify simulation performance and agent behavior.

### Step 1: Implement Metric Function

**File**: `src/metrics.py`

```python
def my_custom_metric(
    reference_prices: Sequence[float],
    simulated_prices: Sequence[float],
    action_log: ActionLog
) -> float:
    """
    Compute your custom metric.
    
    Args:
        reference_prices: Actual market prices
        simulated_prices: Simulated prices from simulator
        action_log: Dict mapping agent names to action lists
        
    Returns:
        Metric value (float)
    """
    if len(reference_prices) != len(simulated_prices):
        raise ValueError("Price series must have equal length")
    
    # Example: Mean absolute percentage error
    mape = np.mean(np.abs(
        (np.array(reference_prices) - np.array(simulated_prices)) / 
        np.array(reference_prices)
    ))
    return float(mape * 100)  # Return as percentage
```

### Step 2: Add to Summary Function

**File**: `src/metrics.py`

```python
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
        # Add your metric here
        "my_custom_metric": my_custom_metric(reference_prices, simulated_prices, action_log),
    }
    return summary
```

### Step 3: Add Tests

**File**: `tests/test_metrics.py` (create if needed)

```python
def test_my_custom_metric():
    """Test custom metric calculation."""
    ref_prices = [100.0, 101.0, 102.0, 103.0]
    sim_prices = [100.0, 100.5, 102.5, 103.0]
    action_log = {"Agent1": ["buy", "hold", "sell", "hold"]}
    
    result = my_custom_metric(ref_prices, sim_prices, action_log)
    
    assert isinstance(result, float)
    assert result >= 0.0  # MAPE is non-negative
```

### Metric Best Practices

1. **Handle Edge Cases**: Empty arrays, single values, all zeros
2. **Return Types**: Consistent return type (float, DataFrame, etc.)
3. **Normalization**: Scale metrics to interpretable ranges
4. **Documentation**: Explain formula and interpretation
5. **Performance**: Optimize for large datasets (vectorize with NumPy)

---

## How to Add New Clustering Algorithms

Currently, the project uses K-Means. You can add DBSCAN, GMM, or custom algorithms.

### Step 1: Implement Clustering Function

**File**: `src/clusterer.py`

```python
from sklearn.cluster import DBSCAN
from typing import Optional

@dataclass
class DBSCANClusterResult:
    """Result from DBSCAN clustering."""
    labels: np.ndarray
    model: DBSCAN
    n_clusters: int
    n_noise: int
    silhouette: Optional[float]

def cluster_embeddings_dbscan(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> DBSCANClusterResult:
    """
    Run DBSCAN on embeddings (density-based clustering).
    
    Args:
        embeddings: 2D array of embeddings
        eps: Maximum distance between samples
        min_samples: Minimum points to form a dense region
        
    Returns:
        DBSCANClusterResult with labels and metadata
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = model.fit_predict(embeddings)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Compute silhouette if we have >1 cluster
    score = None
    if n_clusters > 1:
        # Exclude noise points (-1) from silhouette calculation
        mask = labels != -1
        if mask.sum() > 0:
            from sklearn.metrics import silhouette_score
            score = float(silhouette_score(embeddings[mask], labels[mask]))
    
    return DBSCANClusterResult(
        labels=labels,
        model=model,
        n_clusters=n_clusters,
        n_noise=n_noise,
        silhouette=score
    )
```

### Step 2: Add Tests

**File**: `tests/test_clusterer.py`

```python
def test_cluster_embeddings_dbscan():
    """Test DBSCAN clustering."""
    # Create synthetic embeddings
    np.random.seed(42)
    cluster1 = np.random.randn(30, 10) + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cluster2 = np.random.randn(30, 10) + [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    embeddings = np.vstack([cluster1, cluster2])
    
    result = cluster_embeddings_dbscan(embeddings, eps=1.0, min_samples=5)
    
    assert result.n_clusters >= 2
    assert len(result.labels) == 60
    assert result.silhouette is not None
```

### Step 3: Integrate into Pipeline

**File**: Custom experiment script

```python
from src.clusterer import cluster_embeddings_dbscan

# Generate embeddings
embeddings = embedder.encode_news(df["combined_text"].tolist())

# Use DBSCAN instead of K-Means
cluster_result = cluster_embeddings_dbscan(embeddings, eps=0.5, min_samples=10)
df["cluster_id"] = cluster_result.labels

# Handle noise points (label = -1)
df["cluster_id"] = df["cluster_id"].replace(-1, df["cluster_id"].mode()[0])
```

### Clustering Best Practices

1. **Handle Noise**: Algorithms like DBSCAN output noise (-1); handle gracefully
2. **Hyperparameter Tuning**: Provide sensible defaults
3. **Scalability**: Test on >10,000 samples
4. **Quality Metrics**: Always compute silhouette or similar
5. **Visualization**: Add plotting functions for cluster distributions

---

## How to Integrate New Datasets

Add support for new data sources (e.g., Bloomberg, custom CSVs).

### Step 1: Create Loader Function

**File**: `src/data_loader.py`

```python
def load_my_custom_dataset(
    path: PathLike,
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load custom dataset format.
    
    Args:
        path: Path to data directory or file
        ticker: Optional ticker filter
        start_date: Inclusive start date (YYYY-MM-DD)
        end_date: Inclusive end date (YYYY-MM-DD)
        
    Returns:
        Tuple of (news_df, price_df)
    """
    # Load data (example: JSON, HDF5, API, etc.)
    news_df = pd.read_json(f"{path}/news.json")
    price_df = pd.read_json(f"{path}/prices.json")
    
    # Rename columns to standard format
    news_df = news_df.rename(columns={
        "timestamp": "date",
        "title": "headline",
        "text": "body",
        "symbol": "ticker",
    })
    
    price_df = price_df.rename(columns={
        "timestamp": "date",
        "symbol": "ticker",
    })
    
    # Parse dates
    news_df["date"] = pd.to_datetime(news_df["date"])
    price_df["date"] = pd.to_datetime(price_df["date"])
    
    # Filter by ticker
    if ticker:
        news_df = news_df[news_df["ticker"] == ticker]
        price_df = price_df[price_df["ticker"] == ticker]
    
    # Filter by date range
    if start_date:
        start_ts = pd.to_datetime(start_date)
        news_df = news_df[news_df["date"] >= start_ts]
        price_df = price_df[price_df["date"] >= start_ts]
    
    if end_date:
        end_ts = pd.to_datetime(end_date)
        news_df = news_df[news_df["date"] <= end_ts]
        price_df = price_df[price_df["date"] <= end_ts]
    
    # Ensure required columns
    required_news = {"date", "headline", "ticker"}
    required_price = {"date", "close", "ticker"}
    
    if not required_news.issubset(news_df.columns):
        raise ValueError(f"News data missing columns: {required_news - set(news_df.columns)}")
    
    if not required_price.issubset(price_df.columns):
        raise ValueError(f"Price data missing columns: {required_price - set(price_df.columns)}")
    
    return news_df, price_df
```

### Step 2: Add CLI Support

**File**: `scripts/run_experiment.py`

```python
# Add argument
parser.add_argument("--dataset", choices=["small", "fnspid", "custom"], default="small")
parser.add_argument("--custom-path", type=str, help="Path to custom dataset")

# In main function
if args.dataset == "custom":
    news_df, price_df = load_my_custom_dataset(
        path=args.custom_path,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    merged_df = merge_news_and_prices(news_df, price_df)
```

### Step 3: Document in README

Update `README.md` with usage instructions:

```markdown
### Using Custom Dataset

```bash
python scripts/run_experiment.py \
  --dataset custom \
  --custom-path /path/to/data \
  --ticker AAPL \
  --start-date 2023-01-01
```

### Dataset Best Practices

1. **Standard Schema**: Always output `(news_df, price_df)` with standard columns
2. **Error Handling**: Validate data quality, handle missing values
3. **Documentation**: Explain data format and source
4. **Testing**: Add test with small sample file
5. **Caching**: Cache downloads to avoid repeated API calls

---

## Testing Requirements

All contributions must include tests.

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agents.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Guidelines

1. **Coverage**: Aim for >90% line coverage
2. **Fixtures**: Use pytest fixtures for reusable test data
3. **Parametrize**: Use `@pytest.mark.parametrize` for multiple inputs
4. **Assertions**: Clear, specific assertions with messages
5. **Mocking**: Mock external APIs (Groq, Hugging Face) for speed

### Example Test Fixture

```python
# In tests/conftest.py
import pytest
import pandas as pd

@pytest.fixture
def sample_news_df():
    """Fixture for sample news data."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "headline": ["News 1", "News 2", "News 3"],
        "body": ["Body 1", "Body 2", "Body 3"],
        "ticker": ["AAPL", "AAPL", "AAPL"],
    })

@pytest.fixture
def sample_price_df():
    """Fixture for sample price data."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "close": [150.0, 151.0, 149.0],
        "volume": [1000000, 1100000, 950000],
        "ticker": ["AAPL", "AAPL", "AAPL"],
    })
```

---

## Code Style Guidelines

### Python Style

- **PEP 8**: Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- **Formatter**: Use `black` with default settings
- **Linter**: Use `flake8` (max line length: 120)
- **Type Hints**: Use type annotations for function signatures
- **Docstrings**: Google-style docstrings

### Example

```python
def compute_sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Args:
        returns: Sequence of period returns (e.g., daily)
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Number of periods per year (default: 252 trading days)
        
    Returns:
        Annualized Sharpe ratio
        
    Raises:
        ValueError: If returns are empty or have zero variance
    """
    if not returns:
        raise ValueError("Returns sequence cannot be empty")
    
    ret_array = np.array(returns)
    excess_returns = ret_array - (risk_free_rate / periods_per_year)
    
    if excess_returns.std() == 0:
        raise ValueError("Returns have zero variance")
    
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    return float(sharpe)
```

### Formatting Commands

```bash
# Format code
black src/ tests/ scripts/

# Check lint errors
flake8 src/ tests/ scripts/ --max-line-length=120

# Type checking
mypy src/ --ignore-missing-imports
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add DBSCAN clustering algorithm
fix: handle missing news text in FinBERT agent
docs: update CONTRIBUTING.md with clustering examples
test: add integration test for FNSPID loader
refactor: simplify market state construction
```

---

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write code following style guidelines
- Add tests (aim for >90% coverage)
- Update documentation if needed
- Run `black` and `flake8`

### 3. Commit Changes

```bash
git add .
git commit -m "feat: add my new feature"
```

### 4. Push and Create PR

```bash
git push origin feature/my-new-feature
```

Then create a Pull Request on GitHub with:

**Title**: Clear, concise description (e.g., "Add DBSCAN clustering support")

**Description**:
```markdown
## Summary
Adds DBSCAN clustering as alternative to K-Means.

## Changes
- New function `cluster_embeddings_dbscan()` in `clusterer.py`
- Tests in `test_clusterer.py`
- Example usage in `scripts/run_experiment.py`
- Updated documentation

## Testing
- [x] All tests pass
- [x] Added unit tests for DBSCAN
- [x] Tested on FNSPID dataset

## Checklist
- [x] Code follows style guide
- [x] Tests added and passing
- [x] Documentation updated
- [x] No breaking changes
```

### 5. Code Review

- Address reviewer feedback
- Update PR as needed
- Ensure CI/CD passes

### 6. Merge

Once approved, maintainers will merge your PR.

---

## Community Guidelines

### Code of Conduct

- **Be Respectful**: Treat all contributors with respect
- **Be Inclusive**: Welcome diverse perspectives
- **Be Constructive**: Provide helpful, actionable feedback
- **Be Patient**: Remember everyone is learning

### Getting Help

- **Issues**: Open GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check `docs/` folder for detailed guides

### Reporting Bugs

When reporting bugs, include:

1. **Environment**: OS, Python version, package versions
2. **Steps to Reproduce**: Minimal code example
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Error Messages**: Full traceback if applicable

**Example**:

```markdown
### Bug Report: FinBERT agent crashes on empty news

**Environment**: Windows 10, Python 3.10, transformers 4.30

**Steps to Reproduce**:
```python
from src.agents import FinBERTAgent
agent = FinBERTAgent(name="Test")
action = agent.decide({"price": 100}, cluster_id=None, news_text="")
```

**Expected**: Returns "hold"

**Actual**: Raises `IndexError`

**Traceback**:
```
IndexError: list index out of range
  File "src/agents.py", line 100, in decide
    result = self._pipeline(news_text[:512])[0]
```
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Run tests | `pytest` |
| Format code | `black src/ tests/ scripts/` |
| Lint code | `flake8 src/ --max-line-length=120` |
| Type check | `mypy src/ --ignore-missing-imports` |
| Run experiment | `python scripts/run_experiment.py` |
| Install deps | `pip install -r requirements.txt` |

---

## Additional Resources

- [docs/experimental_design.md](docs/experimental_design.md) - Formal methodology
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) - Reproduction guide
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- [AI_AGENTS_GUIDE.md](AI_AGENTS_GUIDE.md) - AI agent setup
- [README.md](README.md) - Project overview

---

## Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

For questions or suggestions about this guide, please open an issue.

**Happy Coding! 🚀**

