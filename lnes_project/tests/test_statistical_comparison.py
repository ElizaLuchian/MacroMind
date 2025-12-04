"""Tests for statistical testing and agent comparison modules.

This module tests the functionality of statistical_tests.py and agent_comparison.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import statistical_tests, agent_comparison, benchmarks


class TestStatisticalTests:
    """Test suite for statistical_tests module."""
    
    def test_paired_ttest_basic(self):
        """Test basic paired t-test functionality."""
        sample1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample2 = np.array([0.9, 1.8, 2.7, 3.6, 4.5])
        
        result = statistical_tests.paired_ttest(sample1, sample2)
        
        assert "t_statistic" in result
        assert "p_value" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["mean_diff"] > 0  # sample1 > sample2
    
    def test_wilcoxon_test_basic(self):
        """Test Wilcoxon signed-rank test."""
        sample1 = np.array([1, 2, 3, 4, 5])
        sample2 = np.array([1, 2, 3, 4, 4])
        
        result = statistical_tests.wilcoxon_test(sample1, sample2)
        
        assert "statistic" in result
        assert "p_value" in result
        assert "median_diff" in result
    
    def test_cohens_d_effect_size(self):
        """Test Cohen's d effect size calculation."""
        sample1 = np.array([5, 6, 7, 8, 9])
        sample2 = np.array([3, 4, 5, 6, 7])
        
        d = statistical_tests.cohens_d(sample1, sample2)
        
        assert isinstance(d, float)
        assert d > 0  # sample1 > sample2, so positive effect
    
    def test_bootstrap_ci(self):
        """Test bootstrap confidence intervals."""
        sample = np.random.normal(10, 2, 100)
        
        stat, ci_lower, ci_upper = statistical_tests.bootstrap_ci(
            sample,
            statistic_func=np.mean,
            n_bootstrap=1000,
            random_seed=42,
        )
        
        assert ci_lower < stat < ci_upper
        assert ci_lower < 10 < ci_upper  # True mean = 10
    
    def test_permutation_test(self):
        """Test permutation test."""
        sample1 = np.array([5, 6, 7, 8, 9])
        sample2 = np.array([4, 5, 6, 7, 8])
        
        result = statistical_tests.permutation_test(
            sample1,
            sample2,
            n_permutations=1000,
            random_seed=42,
        )
        
        assert "observed_statistic" in result
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1
    
    def test_multiple_comparison_correction(self):
        """Test multiple comparison corrections."""
        p_values = [0.01, 0.04, 0.03, 0.50, 0.008]
        
        for method in ["bonferroni", "holm", "fdr_bh"]:
            result = statistical_tests.multiple_comparison_correction(
                p_values,
                method=method,
                alpha=0.05,
            )
            
            assert len(result) == len(p_values)
            assert "adjusted_p" in result.columns
            assert "reject" in result.columns
            assert all(result["adjusted_p"] >= result["original_p"])


class TestAgentComparison:
    """Test suite for agent_comparison module."""
    
    @pytest.fixture
    def sample_performances(self):
        """Sample agent performance data."""
        np.random.seed(42)
        return {
            "Agent_A": np.random.normal(0.05, 0.02, 100),
            "Agent_B": np.random.normal(0.03, 0.025, 100),
            "Agent_C": np.random.normal(0.04, 0.02, 100),
        }
    
    def test_compare_agents(self, sample_performances):
        """Test comprehensive agent comparison."""
        comparison = agent_comparison.compare_agents(
            sample_performances,
            test_type="ttest",
            correction_method="holm",
        )
        
        assert len(comparison) == 3  # 3 pairwise comparisons
        assert "agent1" in comparison.columns
        assert "agent2" in comparison.columns
        assert "p_value" in comparison.columns
        assert "effect_size" in comparison.columns
        assert "reject" in comparison.columns
    
    def test_rank_agents(self, sample_performances):
        """Test agent ranking with confidence intervals."""
        rankings = agent_comparison.rank_agents(
            sample_performances,
            metric_name="Return",
        )
        
        assert len(rankings) == 3
        assert rankings["rank"].tolist() == [1, 2, 3]
        assert "ci_lower" in rankings.columns
        assert "ci_upper" in rankings.columns
        assert all(rankings["ci_lower"] <= rankings["mean"])
        assert all(rankings["mean"] <= rankings["ci_upper"])
    
    def test_dominance_analysis(self, sample_performances):
        """Test dominance analysis."""
        dominance = agent_comparison.dominance_analysis(sample_performances)
        
        assert isinstance(dominance, dict)
        assert len(dominance) == 3
        for agent, dominated_list in dominance.items():
            assert isinstance(dominated_list, list)
    
    def test_head_to_head(self, sample_performances):
        """Test head-to-head win/loss records."""
        h2h = agent_comparison.head_to_head(sample_performances)
        
        assert len(h2h) == 3  # 3 pairs
        assert "wins" in h2h.columns
        assert "losses" in h2h.columns
        assert "ties" in h2h.columns
        assert "win_rate" in h2h.columns
        assert all(h2h["win_rate"] >= 0)
        assert all(h2h["win_rate"] <= 1)
    
    def test_performance_consistency(self, sample_performances):
        """Test performance consistency analysis."""
        consistency = agent_comparison.performance_consistency(
            sample_performances,
            window_size=10,
        )
        
        assert len(consistency) == 3
        assert "consistency_score" in consistency.columns
        assert all(consistency["consistency_score"] > 0)
        assert all(consistency["consistency_score"] <= 1)
    
    def test_create_comparison_matrix(self, sample_performances):
        """Test comparison matrix creation."""
        matrix = agent_comparison.create_comparison_matrix(
            sample_performances,
            test_type="ttest",
        )
        
        assert matrix.shape == (3, 3)
        assert all(matrix.index == matrix.columns)
        # Diagonal should be "="
        for i in range(3):
            assert matrix.iloc[i, i] == "="


class TestBenchmarks:
    """Test suite for benchmarks module."""
    
    def test_random_benchmark(self):
        """Test random baseline agent."""
        agent = benchmarks.RandomBenchmark(seed=42)
        
        decisions = [agent.decide({"price": 100}) for _ in range(100)]
        
        # Should have all three decision types
        assert "buy" in decisions
        assert "sell" in decisions
        assert "hold" in decisions
    
    def test_buy_and_hold_benchmark(self):
        """Test buy-and-hold baseline."""
        agent = benchmarks.BuyAndHoldBenchmark()
        
        assert agent.decide({"price": 100}) == "buy"
        assert agent.decide({"price": 105}) == "hold"
        assert agent.decide({"price": 95}) == "hold"
    
    def test_momentum_benchmark(self):
        """Test momentum baseline."""
        agent = benchmarks.SimpleMomentumBenchmark()
        
        # Price rising
        agent.decide({"price": 100})
        assert agent.decide({"price": 105}) == "buy"
        
        # Price falling
        assert agent.decide({"price": 103}) == "sell"
    
    def test_benchmark_suite(self):
        """Test benchmark suite collection."""
        suite = benchmarks.BenchmarkSuite(random_seed=42)
        
        all_benchmarks = suite.get_all_benchmarks()
        names = suite.get_benchmark_names()
        
        assert len(all_benchmarks) >= 3
        assert len(names) >= 3
        assert "Random" in names
        assert "BuyAndHold" in names
    
    def test_evaluate_against_benchmarks(self):
        """Test benchmark evaluation."""
        agent_perf = np.random.normal(0.05, 0.02, 100)
        benchmark_perfs = {
            "Random": np.random.normal(0.0, 0.03, 100),
            "BuyAndHold": np.random.normal(0.03, 0.025, 100),
        }
        
        evaluation = benchmarks.evaluate_against_benchmarks(
            agent_perf,
            "TestAgent",
            benchmark_perfs,
        )
        
        assert len(evaluation) == 2
        assert "benchmark" in evaluation.columns
        assert "beats_benchmark" in evaluation.columns
        assert "improvement_pct" in evaluation.columns


class TestIntegration:
    """Integration tests for comparative analysis framework."""
    
    def test_full_comparison_workflow(self):
        """Test complete comparison workflow."""
        np.random.seed(42)
        
        # Generate sample performances
        performances = {
            "Momentum": np.random.normal(0.05, 0.02, 50),
            "Contrarian": np.random.normal(0.03, 0.025, 50),
            "Random": np.random.normal(0.0, 0.03, 50),
        }
        
        # Compare agents
        comparison = agent_comparison.compare_agents(performances)
        assert len(comparison) > 0
        
        # Rank agents
        rankings = agent_comparison.rank_agents(performances)
        assert len(rankings) == 3
        
        # Generate summary report
        report = agent_comparison.summary_report(performances)
        assert "AGENT COMPARISON SUMMARY" in report
        assert "Rankings" in report
        assert "Significant Differences" in report
    
    def test_pairwise_tests_with_correction(self):
        """Test pairwise tests with multiple comparison correction."""
        np.random.seed(42)
        
        performances = {
            f"Agent_{i}": np.random.normal(0.03 + i * 0.01, 0.02, 50)
            for i in range(5)
        }
        
        result = statistical_tests.pairwise_tests(
            performances,
            test_func=statistical_tests.paired_ttest,
            correction_method="holm",
        )
        
        assert len(result) == 10  # 5 choose 2 = 10 pairs
        assert "adjusted_p" in result.columns
        assert "reject" in result.columns

