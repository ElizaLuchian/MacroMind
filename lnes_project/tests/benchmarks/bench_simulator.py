"""Benchmarks for market simulator performance."""

from __future__ import annotations

import pytest

from tests.test_utils import generate_mock_price_data, generate_mock_cluster_labels


# =============================================================================
# Simulator Speed Benchmarks
# =============================================================================

@pytest.mark.benchmark
class TestSimulatorBenchmarks:
    """Benchmark market simulation speed."""
    
    def test_simulator_speed_single_agent(self, benchmark):
        """Benchmark simulation with single agent."""
        from src.simulator import Simulator
        from src.agents import RandomAgent
        
        agent = RandomAgent(seed=42)
        sim = Simulator(agents=[agent], alpha=0.01, noise_std=0.0)
        
        def run_simulation():
            prices = [100.0]
            for i in range(100):
                market_state = {"price": prices[-1], "cluster_id": i % 3}
                action = agent.decide(market_state)
                order_flow = 1 if action == "buy" else (-1 if action == "sell" else 0)
                new_price = max(prices[-1] + 0.01 * order_flow, 0.01)
                prices.append(new_price)
            return prices
        
        result = benchmark(run_simulation)
        assert len(result) == 101
    
    def test_simulator_speed_multiple_agents(self, benchmark):
        """Benchmark simulation with multiple agents."""
        from src.simulator import Simulator
        from src.agents import RandomAgent, MomentumAgent, ContrarianAgent
        
        agents = [
            RandomAgent(seed=42),
            MomentumAgent(),
            ContrarianAgent(),
        ]
        sim = Simulator(agents=agents, alpha=0.01, noise_std=0.0)
        
        def run_simulation():
            prices = [100.0]
            for i in range(100):
                market_state = {"price": prices[-1], "cluster_id": i % 3}
                
                decisions = [agent.decide(market_state) for agent in agents]
                order_flow = sum(
                    1 if d == "buy" else (-1 if d == "sell" else 0)
                    for d in decisions
                )
                
                new_price = max(prices[-1] + 0.01 * order_flow, 0.01)
                prices.append(new_price)
            return prices
        
        result = benchmark(run_simulation)
        assert len(result) == 101
    
    @pytest.mark.parametrize("n_agents", [1, 3, 5, 10])
    def test_simulator_scaling_with_agents(self, benchmark, n_agents):
        """Test how simulation scales with number of agents."""
        from src.simulator import Simulator
        from src.agents import RandomAgent
        
        agents = [RandomAgent(seed=42 + i) for i in range(n_agents)]
        sim = Simulator(agents=agents, alpha=0.01, noise_std=0.0)
        
        def run_simulation():
            prices = [100.0]
            for i in range(50):
                market_state = {"price": prices[-1], "cluster_id": i % 3}
                
                decisions = [agent.decide(market_state) for agent in agents]
                order_flow = sum(
                    1 if d == "buy" else (-1 if d == "sell" else 0)
                    for d in decisions
                )
                
                new_price = max(prices[-1] + 0.01 * order_flow, 0.01)
                prices.append(new_price)
            return prices
        
        result = benchmark(run_simulation)
        assert len(result) == 51


# =============================================================================
# Agent Decision Speed Benchmarks
# =============================================================================

@pytest.mark.benchmark
class TestAgentBenchmarks:
    """Benchmark individual agent decision speed."""
    
    def test_random_agent_speed(self, benchmark):
        """Benchmark Random agent decision speed."""
        from src.agents import RandomAgent
        
        agent = RandomAgent(seed=42)
        market_state = {"price": 100.0, "cluster_id": 0}
        
        def make_decision():
            return agent.decide(market_state)
        
        decision = benchmark(make_decision)
        assert decision in ["buy", "sell", "hold"]
    
    def test_momentum_agent_speed(self, benchmark):
        """Benchmark Momentum agent decision speed."""
        from src.agents import MomentumAgent
        
        agent = MomentumAgent()
        market_state = {"price": 101.0, "prev_price": 100.0}
        
        def make_decision():
            return agent.decide(market_state)
        
        decision = benchmark(make_decision)
        assert decision in ["buy", "sell", "hold"]
    
    def test_contrarian_agent_speed(self, benchmark):
        """Benchmark Contrarian agent decision speed."""
        from src.agents import ContrarianAgent
        
        agent = ContrarianAgent()
        market_state = {"price": 101.0, "prev_price": 100.0}
        
        def make_decision():
            return agent.decide(market_state)
        
        decision = benchmark(make_decision)
        assert decision in ["buy", "sell", "hold"]
    
    def test_news_reactive_agent_speed(self, benchmark):
        """Benchmark NewsReactive agent decision speed."""
        from src.agents import NewsReactiveAgent
        
        cluster_sentiments = {0: 0.5, 1: -0.3, 2: 0.8}
        agent = NewsReactiveAgent(cluster_sentiments=cluster_sentiments)
        market_state = {"price": 100.0, "cluster_id": 0}
        
        def make_decision():
            return agent.decide(market_state)
        
        decision = benchmark(make_decision)
        assert decision in ["buy", "sell", "hold"]


# =============================================================================
# Full Pipeline Benchmarks
# =============================================================================

@pytest.mark.benchmark
class TestPipelineBenchmarks:
    """Benchmark full experiment pipeline."""
    
    def test_small_experiment_speed(self, benchmark):
        """Benchmark small dataset experiment."""
        from src.experiment_smallset import run_small_dataset_experiment
        
        def run_experiment():
            return run_small_dataset_experiment(
                backend="tfidf",
                k_clusters=3,
                agent_types=["Random", "Momentum"],
                alpha=0.01,
                seed=42,
            )
        
        result = benchmark(run_experiment)
        
        assert "metrics" in result
        assert "simulated_prices" in result
    
    @pytest.mark.parametrize("n_clusters", [2, 3, 5, 8])
    def test_clustering_speed_scaling(self, benchmark, n_clusters):
        """Test how clustering scales with k."""
        from src.clusterer import NewsClustering
        from tests.test_utils import generate_mock_embeddings
        
        embeddings = generate_mock_embeddings(n_samples=100, seed=42)
        
        def run_clustering():
            clusterer = NewsClustering(k=n_clusters, random_state=42)
            return clusterer.fit_predict(embeddings)
        
        labels = benchmark(run_clustering)
        assert len(labels) == 100


# =============================================================================
# Metrics Calculation Benchmarks
# =============================================================================

@pytest.mark.benchmark
class TestMetricsBenchmarks:
    """Benchmark metrics calculation speed."""
    
    def test_basic_metrics_speed(self, benchmark):
        """Benchmark basic metrics calculation."""
        from src.metrics import directional_accuracy, final_price_ratio
        
        sim_prices = [100.0 + i * 0.5 for i in range(100)]
        ref_prices = [100.0 + i * 0.6 for i in range(100)]
        
        def calculate_metrics():
            acc = directional_accuracy(sim_prices, ref_prices)
            ratio = final_price_ratio(sim_prices, ref_prices)
            return {"accuracy": acc, "ratio": ratio}
        
        result = benchmark(calculate_metrics)
        assert "accuracy" in result
    
    def test_advanced_metrics_speed(self, benchmark):
        """Benchmark advanced metrics calculation."""
        from src.metrics import (
            sharpe_ratio,
            maximum_drawdown,
            value_at_risk,
        )
        
        prices = [100.0 + i * 0.5 for i in range(100)]
        
        def calculate_advanced_metrics():
            sharpe = sharpe_ratio(prices)
            max_dd = maximum_drawdown(prices)
            var = value_at_risk(prices)
            return {"sharpe": sharpe, "max_dd": max_dd, "var": var}
        
        result = benchmark(calculate_advanced_metrics)
        assert "sharpe" in result
    
    def test_performance_report_speed(self, benchmark):
        """Benchmark full performance report generation."""
        from src.metrics import generate_performance_report
        
        sim_prices = [100.0 + i * 0.5 for i in range(100)]
        ref_prices = [100.0 + i * 0.6 for i in range(100)]
        
        def generate_report():
            return generate_performance_report(
                prices=sim_prices,
                reference_prices=ref_prices,
                risk_free_rate=0.02,
            )
        
        result = benchmark(generate_report)
        assert "directional_accuracy" in result


# =============================================================================
# Memory Efficiency Benchmarks
# =============================================================================

@pytest.mark.benchmark
class TestMemoryBenchmarks:
    """Test memory efficiency of operations."""
    
    def test_simulation_memory_efficiency(self):
        """Test simulation doesn't consume excessive memory."""
        from src.simulator import Simulator
        from src.agents import RandomAgent, MomentumAgent
        
        agents = [RandomAgent(seed=42), MomentumAgent()]
        sim = Simulator(agents=agents, alpha=0.01)
        
        # Run long simulation
        prices = [100.0]
        action_log = {agent.name: [] for agent in agents}
        
        for i in range(1000):
            market_state = {"price": prices[-1], "cluster_id": i % 3}
            
            for agent in agents:
                action = agent.decide(market_state)
                action_log[agent.name].append(action)
            
            decisions = [action_log[agent.name][-1] for agent in agents]
            order_flow = sum(
                1 if d == "buy" else (-1 if d == "sell" else 0)
                for d in decisions
            )
            
            new_price = max(prices[-1] + 0.01 * order_flow, 0.01)
            prices.append(new_price)
        
        # Check results
        assert len(prices) == 1001
        assert all(len(actions) == 1000 for actions in action_log.values())
    
    def test_metrics_memory_efficiency(self):
        """Test metrics calculation doesn't use excessive memory."""
        from src.metrics import generate_performance_report
        
        # Large price series
        prices = [100.0 + i * 0.1 for i in range(10000)]
        ref_prices = [100.0 + i * 0.12 for i in range(10000)]
        
        report = generate_performance_report(
            prices=prices,
            reference_prices=ref_prices,
        )
        
        # Should complete without memory issues
        assert "directional_accuracy" in report


# =============================================================================
# Regression Detection Benchmarks
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.regression
class TestRegressionBenchmarks:
    """Benchmarks for detecting performance regressions."""
    
    def test_baseline_simulation_performance(self, benchmark):
        """Establish baseline simulation performance."""
        from src.simulator import Simulator
        from src.agents import RandomAgent, MomentumAgent
        
        agents = [RandomAgent(seed=42), MomentumAgent()]
        sim = Simulator(agents=agents, alpha=0.01)
        
        def run_simulation():
            prices = [100.0]
            for i in range(100):
                market_state = {"price": prices[-1], "cluster_id": i % 3}
                
                decisions = [agent.decide(market_state) for agent in agents]
                order_flow = sum(
                    1 if d == "buy" else (-1 if d == "sell" else 0)
                    for d in decisions
                )
                
                new_price = max(prices[-1] + 0.01 * order_flow, 0.01)
                prices.append(new_price)
            return prices
        
        result = benchmark(run_simulation)
        
        # Should complete quickly (< 0.1 seconds for 100 steps)
        stats = benchmark.stats
        assert stats.mean < 0.1
        
        assert len(result) == 101
    
    def test_baseline_metrics_performance(self, benchmark):
        """Establish baseline metrics calculation performance."""
        from src.metrics import generate_performance_report
        
        prices = [100.0 + i * 0.5 for i in range(100)]
        ref_prices = [100.0 + i * 0.6 for i in range(100)]
        
        def calculate_metrics():
            return generate_performance_report(
                prices=prices,
                reference_prices=ref_prices,
            )
        
        result = benchmark(calculate_metrics)
        
        # Should be fast (< 0.05 seconds)
        stats = benchmark.stats
        assert stats.mean < 0.05
        
        assert "directional_accuracy" in result


# =============================================================================
# Comparative Benchmarks
# =============================================================================

@pytest.mark.benchmark
class TestComparativeBenchmarks:
    """Compare performance of different implementations."""
    
    def test_numpy_vs_list_performance(self, benchmark):
        """Compare numpy vs list for price updates."""
        import numpy as np
        
        prices_list = [100.0]
        
        def update_with_list():
            prices = [100.0]
            for i in range(100):
                new_price = prices[-1] + 0.1
                prices.append(new_price)
            return prices
        
        result = benchmark(update_with_list)
        assert len(result) == 101
    
    def test_agent_decision_overhead(self, benchmark):
        """Measure overhead of agent decision-making."""
        from src.agents import RandomAgent
        
        agent = RandomAgent(seed=42)
        
        def make_many_decisions():
            decisions = []
            for i in range(1000):
                market_state = {"price": 100.0, "cluster_id": i % 3}
                decision = agent.decide(market_state)
                decisions.append(decision)
            return decisions
        
        result = benchmark(make_many_decisions)
        assert len(result) == 1000


# =============================================================================
# Stress Tests
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.slow
class TestStressBenchmarks:
    """Stress tests for extreme scenarios."""
    
    def test_many_agents_simulation(self, benchmark):
        """Test simulation with many agents."""
        from src.simulator import Simulator
        from src.agents import RandomAgent
        
        # 50 agents
        agents = [RandomAgent(seed=42 + i) for i in range(50)]
        sim = Simulator(agents=agents, alpha=0.01)
        
        def run_simulation():
            prices = [100.0]
            for i in range(50):
                market_state = {"price": prices[-1], "cluster_id": i % 3}
                
                decisions = [agent.decide(market_state) for agent in agents]
                order_flow = sum(
                    1 if d == "buy" else (-1 if d == "sell" else 0)
                    for d in decisions
                )
                
                new_price = max(prices[-1] + 0.01 * order_flow, 0.01)
                prices.append(new_price)
            return prices
        
        result = benchmark(run_simulation)
        assert len(result) == 51
    
    def test_long_simulation(self, benchmark):
        """Test very long simulation."""
        from src.simulator import Simulator
        from src.agents import RandomAgent
        
        agent = RandomAgent(seed=42)
        sim = Simulator(agents=[agent], alpha=0.01)
        
        def run_simulation():
            prices = [100.0]
            for i in range(1000):
                market_state = {"price": prices[-1], "cluster_id": i % 3}
                action = agent.decide(market_state)
                order_flow = 1 if action == "buy" else (-1 if action == "sell" else 0)
                new_price = max(prices[-1] + 0.01 * order_flow, 0.01)
                prices.append(new_price)
            return prices
        
        result = benchmark(run_simulation)
        assert len(result) == 1001

