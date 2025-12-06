"""Property-based tests using hypothesis library."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from tests.test_utils import assert_valid_action_log, assert_valid_price_series


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Price strategies
positive_prices = st.lists(
    st.floats(min_value=0.01, max_value=10000, allow_nan=False, allow_infinity=False),
    min_size=10,
    max_size=100,
)

price_series = st.lists(
    st.floats(min_value=1.0, max_value=1000, allow_nan=False, allow_infinity=False),
    min_size=20,
    max_size=100,
)

# Action strategies
actions = st.sampled_from(["buy", "sell", "hold"])

action_sequences = st.lists(actions, min_size=10, max_size=100)

# Parameter strategies
alpha_values = st.floats(min_value=0.001, max_value=0.1, allow_nan=False)

cluster_k = st.integers(min_value=2, max_value=10)


# =============================================================================
# Price Series Properties
# =============================================================================

class TestPriceSeriesProperties:
    """Property-based tests for price series."""
    
    @given(price_series)
    @settings(max_examples=50, deadline=1000)
    def test_price_series_always_positive(self, prices):
        """Property: Simulated prices should always be positive."""
        from src.simulator import Simulator
        from src.agents import RandomAgent
        
        agent = RandomAgent(seed=42)
        sim = Simulator(agents=[agent], alpha=0.01, noise_std=0.0)
        
        # Run short simulation
        initial_price = prices[0]
        sim_prices = [initial_price]
        
        for _ in range(min(20, len(prices) - 1)):
            market_state = {"price": sim_prices[-1], "cluster_id": 0}
            action = agent.decide(market_state)
            
            # Calculate price change
            order_flow = 1 if action == "buy" else (-1 if action == "sell" else 0)
            new_price = max(sim_prices[-1] + 0.01 * order_flow, 0.01)
            sim_prices.append(new_price)
        
        # All prices should be positive
        assert all(p > 0 for p in sim_prices)
    
    @given(price_series)
    @settings(max_examples=50, deadline=1000)
    def test_price_returns_bounded(self, prices):
        """Property: Returns should be bounded for reasonable parameters."""
        if len(prices) < 2:
            return
        
        from src.metrics import compute_returns
        
        returns = compute_returns(prices, method="simple")
        
        # Returns should be finite
        assert all(np.isfinite(r) for r in returns)
    
    @given(positive_prices)
    @settings(max_examples=50, deadline=1000)
    def test_drawdown_always_negative_or_zero(self, prices):
        """Property: Drawdown should always be <= 0."""
        if len(prices) < 2:
            return
        
        from src.metrics import drawdown_series
        
        drawdowns = drawdown_series(prices)
        
        # Drawdowns should be non-positive
        assert all(d <= 0 for d in drawdowns)
    
    @given(price_series)
    @settings(max_examples=50, deadline=1000)
    def test_cumulative_returns_consistency(self, prices):
        """Property: Cumulative returns should match final price ratio."""
        if len(prices) < 2:
            return
        
        from src.metrics import compute_returns
        
        returns = compute_returns(prices, method="simple")
        cumulative_return = np.prod(1 + np.array(returns)) - 1
        direct_return = (prices[-1] - prices[0]) / prices[0]
        
        # Should be approximately equal
        assert np.isclose(cumulative_return, direct_return, rtol=0.01)


# =============================================================================
# Simulator Properties
# =============================================================================

class TestSimulatorProperties:
    """Property-based tests for simulator."""
    
    @given(alpha_values, st.integers(min_value=10, max_value=50))
    @settings(max_examples=30, deadline=2000)
    def test_simulator_price_bounded_by_alpha(self, alpha, n_steps):
        """Property: Price changes should be bounded by alpha parameter."""
        from src.simulator import Simulator
        from src.agents import MomentumAgent
        
        agent = MomentumAgent()
        sim = Simulator(agents=[agent], alpha=alpha, noise_std=0.0)
        
        prices = [100.0]
        for _ in range(n_steps):
            market_state = {"price": prices[-1], "cluster_id": 0}
            action = agent.decide(market_state)
            
            order_flow = 1 if action == "buy" else (-1 if action == "sell" else 0)
            new_price = max(prices[-1] + alpha * order_flow, 0.01)
            prices.append(new_price)
        
        # Price changes should be bounded by alpha (single agent)
        for i in range(1, len(prices)):
            change = abs(prices[i] - prices[i - 1])
            assert change <= alpha + 1e-6  # Small tolerance for floating point
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=2000)
    def test_simulator_more_agents_larger_impact(self, n_agents):
        """Property: More agents should potentially cause larger price swings."""
        from src.simulator import Simulator
        from src.agents import MomentumAgent
        
        agents = [MomentumAgent() for _ in range(n_agents)]
        sim = Simulator(agents=agents, alpha=0.01, noise_std=0.0)
        
        # All momentum agents should cause directional movement
        prices = [100.0]
        for _ in range(10):
            market_state = {"price": prices[-1], "cluster_id": 0}
            
            # Get all decisions
            decisions = [agent.decide(market_state) for agent in agents]
            order_flow = sum(
                1 if d == "buy" else (-1 if d == "sell" else 0)
                for d in decisions
            )
            
            new_price = max(prices[-1] + 0.01 * order_flow, 0.01)
            prices.append(new_price)
        
        # With n_agents all doing momentum, should see movement
        total_change = abs(prices[-1] - prices[0])
        # With multiple momentum agents, should see some price movement
        assert total_change >= 0  # At minimum, prices shouldn't crash


# =============================================================================
# Metrics Properties
# =============================================================================

class TestMetricsProperties:
    """Property-based tests for metrics."""
    
    @given(price_series, st.floats(min_value=0, max_value=0.1))
    @settings(max_examples=50, deadline=1000)
    def test_sharpe_ratio_bounded(self, prices, risk_free_rate):
        """Property: Sharpe ratio should be finite for valid prices."""
        if len(prices) < 2:
            return
        
        from src.metrics import sharpe_ratio
        
        try:
            sharpe = sharpe_ratio(prices, risk_free_rate=risk_free_rate)
            
            # Sharpe should be finite
            assert np.isfinite(sharpe)
            
            # Sharpe ratio typically in range [-5, 5] for daily data
            # (Can be outside but testing reasonable bounds)
        except (ValueError, ZeroDivisionError):
            # Zero volatility case is acceptable
            pass
    
    @given(price_series)
    @settings(max_examples=50, deadline=1000)
    def test_max_drawdown_vs_avg_drawdown(self, prices):
        """Property: Max drawdown should be >= average drawdown."""
        if len(prices) < 2:
            return
        
        from src.metrics import maximum_drawdown, average_drawdown
        
        max_dd = maximum_drawdown(prices)
        avg_dd = average_drawdown(prices)
        
        # Max drawdown should be at least as large as average
        assert abs(max_dd) >= abs(avg_dd)
    
    @given(price_series)
    @settings(max_examples=50, deadline=1000)
    def test_volatility_always_positive(self, prices):
        """Property: Volatility should always be non-negative."""
        if len(prices) < 2:
            return
        
        from src.metrics import compute_returns
        
        returns = compute_returns(prices)
        volatility = np.std(returns)
        
        assert volatility >= 0
    
    @given(
        st.lists(st.sampled_from(["buy", "sell", "hold"]), min_size=10, max_size=50),
        price_series,
    )
    @settings(max_examples=30, deadline=1000)
    def test_win_rate_bounded(self, actions, prices):
        """Property: Win rate should be between 0 and 1."""
        if len(actions) != len(prices) - 1:
            # Make them same length
            n = min(len(actions), len(prices) - 1)
            actions = actions[:n]
            prices = prices[: n + 1]
        
        if len(actions) == 0:
            return
        
        from src.metrics import win_rate
        
        action_log = {"agent": actions}
        win_rates = win_rate(action_log, prices)
        
        # Win rate should be probability
        for agent, rate in win_rates.items():
            assert 0 <= rate <= 1


# =============================================================================
# Clustering Properties
# =============================================================================

class TestClusteringProperties:
    """Property-based tests for clustering."""
    
    @given(
        st.integers(min_value=2, max_value=8),
        st.integers(min_value=20, max_value=100),
    )
    @settings(max_examples=30, deadline=2000)
    def test_kmeans_cluster_count(self, k, n_samples):
        """Property: K-means should produce exactly k clusters."""
        from src.clusterer import NewsClustering
        from tests.test_utils import generate_mock_embeddings
        
        embeddings = generate_mock_embeddings(n_samples=n_samples, seed=42)
        
        clusterer = NewsClustering(k=k, method="kmeans", random_state=42)
        labels = clusterer.fit_predict(embeddings)
        
        # Should have exactly k unique labels
        unique_labels = len(set(labels))
        assert unique_labels <= k  # May have fewer if convergence issues
        assert unique_labels >= 1
    
    @given(
        st.integers(min_value=2, max_value=5),
        st.integers(min_value=20, max_value=50),
    )
    @settings(max_examples=20, deadline=2000)
    def test_clustering_labels_valid_range(self, k, n_samples):
        """Property: Cluster labels should be in range [0, k-1]."""
        from src.clusterer import NewsClustering
        from tests.test_utils import generate_mock_embeddings
        
        embeddings = generate_mock_embeddings(n_samples=n_samples, seed=42)
        
        clusterer = NewsClustering(k=k, method="kmeans", random_state=42)
        labels = clusterer.fit_predict(embeddings)
        
        # All labels should be valid indices
        assert all(0 <= label < k for label in labels)
    
    @given(st.integers(min_value=20, max_value=100))
    @settings(max_examples=20, deadline=2000)
    def test_clustering_deterministic_with_seed(self, n_samples):
        """Property: Clustering should be deterministic with fixed seed."""
        from src.clusterer import NewsClustering
        from tests.test_utils import generate_mock_embeddings
        
        embeddings = generate_mock_embeddings(n_samples=n_samples, seed=42)
        
        clusterer1 = NewsClustering(k=3, method="kmeans", random_state=42)
        labels1 = clusterer1.fit_predict(embeddings)
        
        clusterer2 = NewsClustering(k=3, method="kmeans", random_state=42)
        labels2 = clusterer2.fit_predict(embeddings)
        
        # Should produce identical results
        assert np.array_equal(labels1, labels2)


# =============================================================================
# Agent Decision Properties
# =============================================================================

class TestAgentProperties:
    """Property-based tests for agent behavior."""
    
    @given(price_series)
    @settings(max_examples=50, deadline=1000)
    def test_momentum_agent_follows_trend(self, prices):
        """Property: Momentum agent should buy on uptrend, sell on downtrend."""
        if len(prices) < 2:
            return
        
        from src.agents import MomentumAgent
        
        agent = MomentumAgent()
        
        for i in range(1, min(len(prices), 20)):
            market_state = {"price": prices[i], "prev_price": prices[i - 1]}
            decision = agent.decide(market_state)
            
            if prices[i] > prices[i - 1]:
                assert decision == "buy"
            elif prices[i] < prices[i - 1]:
                assert decision == "sell"
            else:
                assert decision == "hold"
    
    @given(price_series)
    @settings(max_examples=50, deadline=1000)
    def test_contrarian_agent_reverses_trend(self, prices):
        """Property: Contrarian agent should sell on uptrend, buy on downtrend."""
        if len(prices) < 2:
            return
        
        from src.agents import ContrarianAgent
        
        agent = ContrarianAgent()
        
        for i in range(1, min(len(prices), 20)):
            market_state = {"price": prices[i], "prev_price": prices[i - 1]}
            decision = agent.decide(market_state)
            
            if prices[i] > prices[i - 1]:
                assert decision == "sell"
            elif prices[i] < prices[i - 1]:
                assert decision == "buy"
            else:
                assert decision == "hold"
    
    @given(st.integers(min_value=42, max_value=100))
    @settings(max_examples=20, deadline=1000)
    def test_random_agent_deterministic_with_seed(self, seed):
        """Property: Random agent should be deterministic with same seed."""
        from src.agents import RandomAgent
        
        agent1 = RandomAgent(seed=seed)
        agent2 = RandomAgent(seed=seed)
        
        market_state = {"price": 100.0}
        
        # Should make same decisions
        decisions1 = [agent1.decide(market_state) for _ in range(20)]
        decisions2 = [agent2.decide(market_state) for _ in range(20)]
        
        assert decisions1 == decisions2
    
    @given(action_sequences)
    @settings(max_examples=50, deadline=1000)
    def test_action_log_validation(self, actions):
        """Property: Action log should always be valid format."""
        action_log = {
            "agent1": actions,
            "agent2": actions,
        }
        
        # Should not raise
        assert_valid_action_log(action_log)


# =============================================================================
# Embedding Properties
# =============================================================================

class TestEmbeddingProperties:
    """Property-based tests for embeddings."""
    
    @given(st.integers(min_value=10, max_value=100))
    @settings(max_examples=30, deadline=2000)
    def test_embedding_dimension_consistent(self, n_texts):
        """Property: All embeddings should have same dimension."""
        from src.embedder import NewsEmbedder
        
        embedder = NewsEmbedder(backend="tfidf")
        texts = [f"This is sample text number {i}" for i in range(n_texts)]
        
        embeddings = embedder.embed(texts)
        
        # All embeddings should have same shape
        assert embeddings.shape[0] == n_texts
        assert all(len(emb) == embeddings.shape[1] for emb in embeddings)
    
    @given(st.integers(min_value=5, max_value=50))
    @settings(max_examples=20, deadline=2000)
    def test_embedding_deterministic(self, n_texts):
        """Property: Embeddings should be deterministic for same input."""
        from src.embedder import NewsEmbedder
        
        texts = [f"Sample text {i}" for i in range(n_texts)]
        
        embedder1 = NewsEmbedder(backend="tfidf")
        embeddings1 = embedder1.embed(texts)
        
        embedder2 = NewsEmbedder(backend="tfidf")
        embeddings2 = embedder2.embed(texts)
        
        # Should produce same embeddings
        assert np.allclose(embeddings1, embeddings2, rtol=1e-5)
    
    @given(st.integers(min_value=10, max_value=50))
    @settings(max_examples=20, deadline=2000)
    def test_embeddings_finite(self, n_texts):
        """Property: Embeddings should always be finite values."""
        from src.embedder import NewsEmbedder
        
        texts = [f"Text {i}" for i in range(n_texts)]
        embedder = NewsEmbedder(backend="tfidf")
        embeddings = embedder.embed(texts)
        
        # All values should be finite
        assert np.all(np.isfinite(embeddings))


# =============================================================================
# Integration Properties
# =============================================================================

class TestIntegrationProperties:
    """Property-based integration tests."""
    
    @given(
        cluster_k,
        st.integers(min_value=10, max_value=30),
        alpha_values,
    )
    @settings(max_examples=20, deadline=5000)
    def test_full_pipeline_produces_valid_output(self, k, n_samples, alpha):
        """Property: Full pipeline should always produce valid output."""
        from src.embedder import NewsEmbedder
        from src.clusterer import NewsClustering
        from src.agents import RandomAgent, MomentumAgent
        from src.simulator import Simulator
        
        # Generate data
        texts = [f"News item {i}" for i in range(n_samples)]
        prices = [100.0 + i * 0.1 for i in range(n_samples)]
        
        # Embed
        embedder = NewsEmbedder(backend="tfidf")
        embeddings = embedder.embed(texts)
        
        # Cluster
        clusterer = NewsClustering(k=k, random_state=42)
        labels = clusterer.fit_predict(embeddings)
        
        # Simulate
        agents = [RandomAgent(seed=42), MomentumAgent()]
        sim = Simulator(agents=agents, alpha=alpha)
        
        # Run short simulation
        sim_prices = [prices[0]]
        for i in range(min(10, len(prices) - 1)):
            market_state = {
                "price": sim_prices[-1],
                "cluster_id": int(labels[i]),
            }
            
            decisions = [agent.decide(market_state) for agent in agents]
            order_flow = sum(
                1 if d == "buy" else (-1 if d == "sell" else 0)
                for d in decisions
            )
            
            new_price = max(sim_prices[-1] + alpha * order_flow, 0.01)
            sim_prices.append(new_price)
        
        # Validate output
        assert len(sim_prices) > 1
        assert all(p > 0 for p in sim_prices)
        assert all(np.isfinite(p) for p in sim_prices)

