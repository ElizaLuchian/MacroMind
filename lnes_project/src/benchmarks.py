"""Benchmark agent implementations for comparison studies.

This module provides standard baseline agents for comparing experimental agents against
established benchmarks. These include random trading, buy-and-hold, momentum strategies,
and mean-reversion strategies.

Classes:
    RandomBenchmark: Random action selection
    BuyAndHoldBenchmark: Always buy, never sell
    SimpleMomentumBenchmark: Buy on uptrend, sell on downtrend
    SimpleMeanReversionBenchmark: Buy on dips, sell on peaks
    MovingAverageCrossover: MA crossover strategy
    BenchmarkSuite: Collection of all benchmark agents
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    """Container for benchmark performance results."""
    agent_name: str
    final_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int


class RandomBenchmark:
    """Random baseline agent - selects actions uniformly at random.
    
    Expected return: 0 (over many trials)
    Expected win rate: ~33% (if 3 actions)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize random benchmark.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.name = "Random"
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def decide(self, market_state: Dict[str, Any]) -> str:
        """Make random decision.
        
        Args:
            market_state: Current market state (ignored)
            
        Returns:
            Random action: "buy", "sell", or "hold"
        """
        return self.rng.choice(["buy", "sell", "hold"])
    
    def reset(self):
        """Reset internal state."""
        if self.seed is not None:
            self.rng = np.random.RandomState(self.seed)


class BuyAndHoldBenchmark:
    """Buy-and-hold baseline - always buys, never sells.
    
    Expected return: Equals market return
    Win rate: 100% on uptrends, 0% on downtrends
    """
    
    def __init__(self):
        """Initialize buy-and-hold benchmark."""
        self.name = "BuyAndHold"
        self.has_bought = False
    
    def decide(self, market_state: Dict[str, Any]) -> str:
        """Always buy (once), then hold.
        
        Args:
            market_state: Current market state
            
        Returns:
            "buy" on first call, "hold" afterwards
        """
        if not self.has_bought:
            self.has_bought = True
            return "buy"
        return "hold"
    
    def reset(self):
        """Reset internal state."""
        self.has_bought = False


class SimpleMomentumBenchmark:
    """Simple momentum baseline - follows price trends.
    
    Strategy:
        - Buy if price is rising
        - Sell if price is falling
        - Hold if price unchanged
    
    Expected win rate: >50% in trending markets, <50% in mean-reverting markets
    """
    
    def __init__(self):
        """Initialize momentum benchmark."""
        self.name = "SimpleMomentum"
        self.prev_price = None
    
    def decide(self, market_state: Dict[str, Any]) -> str:
        """Decide based on price momentum.
        
        Args:
            market_state: Must contain "price" key
            
        Returns:
            "buy" if price rising, "sell" if falling, "hold" if unchanged
        """
        current_price = market_state.get("price")
        
        if current_price is None:
            return "hold"
        
        if self.prev_price is None:
            self.prev_price = current_price
            return "hold"
        
        if current_price > self.prev_price:
            action = "buy"
        elif current_price < self.prev_price:
            action = "sell"
        else:
            action = "hold"
        
        self.prev_price = current_price
        return action
    
    def reset(self):
        """Reset internal state."""
        self.prev_price = None


class SimpleMeanReversionBenchmark:
    """Simple mean-reversion baseline - bets on price reversals.
    
    Strategy:
        - Sell if price is rising (expect reversal)
        - Buy if price is falling (expect bounce)
        - Hold if price unchanged
    
    Expected win rate: >50% in range-bound markets, <50% in trending markets
    """
    
    def __init__(self):
        """Initialize mean-reversion benchmark."""
        self.name = "SimpleMeanReversion"
        self.prev_price = None
    
    def decide(self, market_state: Dict[str, Any]) -> str:
        """Decide based on mean reversion.
        
        Args:
            market_state: Must contain "price" key
            
        Returns:
            "sell" if price rising, "buy" if falling, "hold" if unchanged
        """
        current_price = market_state.get("price")
        
        if current_price is None:
            return "hold"
        
        if self.prev_price is None:
            self.prev_price = current_price
            return "hold"
        
        if current_price > self.prev_price:
            action = "sell"  # Expect reversal
        elif current_price < self.prev_price:
            action = "buy"  # Expect bounce
        else:
            action = "hold"
        
        self.prev_price = current_price
        return action
    
    def reset(self):
        """Reset internal state."""
        self.prev_price = None


class MovingAverageCrossover:
    """Moving average crossover strategy.
    
    Strategy:
        - Buy when short MA crosses above long MA (golden cross)
        - Sell when short MA crosses below long MA (death cross)
        - Hold otherwise
    
    Common parameters:
        - short_window: 20 days
        - long_window: 50 days
    """
    
    def __init__(self, short_window: int = 5, long_window: int = 10):
        """Initialize MA crossover benchmark.
        
        Args:
            short_window: Short moving average window
            long_window: Long moving average window
        """
        self.name = f"MA_Crossover_{short_window}_{long_window}"
        self.short_window = short_window
        self.long_window = long_window
        self.price_history = []
        self.prev_short_ma = None
        self.prev_long_ma = None
    
    def decide(self, market_state: Dict[str, Any]) -> str:
        """Decide based on MA crossover.
        
        Args:
            market_state: Must contain "price" key
            
        Returns:
            "buy" on golden cross, "sell" on death cross, "hold" otherwise
        """
        current_price = market_state.get("price")
        
        if current_price is None:
            return "hold"
        
        self.price_history.append(current_price)
        
        # Need enough history
        if len(self.price_history) < self.long_window:
            return "hold"
        
        # Compute MAs
        recent_prices = np.array(self.price_history[-self.long_window:])
        short_ma = np.mean(recent_prices[-self.short_window:])
        long_ma = np.mean(recent_prices)
        
        # Detect crossover
        action = "hold"
        if self.prev_short_ma is not None and self.prev_long_ma is not None:
            # Golden cross: short MA crosses above long MA
            if self.prev_short_ma <= self.prev_long_ma and short_ma > long_ma:
                action = "buy"
            # Death cross: short MA crosses below long MA
            elif self.prev_short_ma >= self.prev_long_ma and short_ma < long_ma:
                action = "sell"
        
        self.prev_short_ma = short_ma
        self.prev_long_ma = long_ma
        
        return action
    
    def reset(self):
        """Reset internal state."""
        self.price_history = []
        self.prev_short_ma = None
        self.prev_long_ma = None


class BenchmarkSuite:
    """Collection of benchmark agents for systematic comparison.
    
    Usage:
        suite = BenchmarkSuite()
        agents = suite.get_all_benchmarks()
        for agent in agents:
            # Run simulation with agent
            ...
    """
    
    def __init__(self, random_seed: Optional[int] = 42):
        """Initialize benchmark suite.
        
        Args:
            random_seed: Seed for random benchmark
        """
        self.random_seed = random_seed
        self.benchmarks = {
            "Random": RandomBenchmark(seed=random_seed),
            "BuyAndHold": BuyAndHoldBenchmark(),
            "SimpleMomentum": SimpleMomentumBenchmark(),
            "SimpleMeanReversion": SimpleMeanReversionBenchmark(),
            "MA_Crossover_5_10": MovingAverageCrossover(short_window=5, long_window=10),
        }
    
    def get_benchmark(self, name: str):
        """Get a specific benchmark agent.
        
        Args:
            name: Benchmark name
            
        Returns:
            Benchmark agent instance
        """
        if name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {name}. Available: {list(self.benchmarks.keys())}")
        return self.benchmarks[name]
    
    def get_all_benchmarks(self) -> List:
        """Get all benchmark agents.
        
        Returns:
            List of benchmark agent instances
        """
        return list(self.benchmarks.values())
    
    def get_benchmark_names(self) -> List[str]:
        """Get names of all benchmarks.
        
        Returns:
            List of benchmark names
        """
        return list(self.benchmarks.keys())
    
    def reset_all(self):
        """Reset all benchmarks to initial state."""
        for benchmark in self.benchmarks.values():
            benchmark.reset()


def evaluate_against_benchmarks(
    agent_performance: np.ndarray,
    agent_name: str,
    benchmark_performances: Dict[str, np.ndarray],
    metric_name: str = "Return",
) -> pd.DataFrame:
    """Evaluate an agent's performance against benchmarks.
    
    Args:
        agent_performance: Agent's performance array
        agent_name: Name of the agent
        benchmark_performances: Dict mapping benchmark names to performance arrays
        metric_name: Name of the performance metric
        
    Returns:
        DataFrame with comparison results:
            - benchmark: benchmark name
            - agent_mean: agent's mean performance
            - benchmark_mean: benchmark's mean performance
            - difference: agent_mean - benchmark_mean
            - improvement_pct: percentage improvement over benchmark
            - beats_benchmark: whether agent beats benchmark
            
    Example:
        >>> results = evaluate_against_benchmarks(
        ...     my_agent_returns,
        ...     "MyAgent",
        ...     {"Random": random_returns, "BuyAndHold": bh_returns}
        ... )
        >>> print(results)
    """
    agent_mean = np.mean(agent_performance)
    
    results = []
    for bench_name, bench_performance in benchmark_performances.items():
        bench_mean = np.mean(bench_performance)
        diff = agent_mean - bench_mean
        
        # Percentage improvement
        if abs(bench_mean) > 1e-10:
            improvement_pct = (diff / abs(bench_mean)) * 100
        else:
            improvement_pct = 0.0 if abs(diff) < 1e-10 else np.inf
        
        results.append({
            "benchmark": bench_name,
            "agent_mean": agent_mean,
            "benchmark_mean": bench_mean,
            "difference": diff,
            "improvement_pct": improvement_pct,
            "beats_benchmark": diff > 0,
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("difference", ascending=False).reset_index(drop=True)
    
    return df


def benchmark_threshold_analysis(
    performances: Dict[str, np.ndarray],
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Analyze which agents meet performance thresholds.
    
    Args:
        performances: Dict mapping agent names to performance arrays
        thresholds: Dict mapping metric names to threshold values
            Default: {"mean": 0.0, "sharpe": 0.5, "win_rate": 0.5}
            
    Returns:
        DataFrame showing which agents meet which thresholds
        
    Example:
        >>> perfs = {"AgentA": returns_a, "AgentB": returns_b}
        >>> thresholds = {"mean": 0.01, "sharpe": 1.0}
        >>> results = benchmark_threshold_analysis(perfs, thresholds)
        >>> print(results)
    """
    if thresholds is None:
        thresholds = {
            "mean": 0.0,
            "sharpe": 0.5,
            "win_rate": 0.5,
        }
    
    results = []
    
    for agent_name, performance in performances.items():
        mean_perf = np.mean(performance)
        std_perf = np.std(performance, ddof=1)
        sharpe = mean_perf / std_perf if std_perf > 0 else 0.0
        win_rate = np.mean(performance > 0)
        
        metrics = {
            "mean": mean_perf,
            "sharpe": sharpe,
            "win_rate": win_rate,
        }
        
        row = {"agent": agent_name}
        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                meets_threshold = value >= threshold
                row[f"{metric_name}_value"] = value
                row[f"{metric_name}_threshold"] = threshold
                row[f"meets_{metric_name}"] = meets_threshold
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_benchmark_statistics(
    benchmark_performances: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute summary statistics for all benchmarks.
    
    Args:
        benchmark_performances: Dict mapping benchmark names to performance arrays
        
    Returns:
        DataFrame with statistics for each benchmark:
            - benchmark: name
            - mean, std, min, max, median
            - sharpe_ratio
            - win_rate
            
    Example:
        >>> stats = compute_benchmark_statistics(benchmark_perfs)
        >>> print(stats)
    """
    results = []
    
    for bench_name, performance in benchmark_performances.items():
        mean_perf = np.mean(performance)
        std_perf = np.std(performance, ddof=1)
        sharpe = mean_perf / std_perf if std_perf > 0 else 0.0
        win_rate = np.mean(performance > 0)
        
        results.append({
            "benchmark": bench_name,
            "mean": mean_perf,
            "std": std_perf,
            "min": np.min(performance),
            "max": np.max(performance),
            "median": np.median(performance),
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "n_samples": len(performance),
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    
    return df

