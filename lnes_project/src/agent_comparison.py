"""Agent comparison framework for systematic performance analysis.

This module provides tools for comparing trading agent performance using
statistical tests, ranking methods, and dominance analysis.

Functions:
    compare_agents: Comprehensive pairwise comparison
    rank_agents: Rank agents with confidence intervals
    dominance_analysis: Identify dominant strategies
    head_to_head: Win/loss/tie records between agents
    performance_consistency: Analyze performance stability
    pareto_frontier: Identify Pareto-optimal agents
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from .statistical_tests import (
    paired_ttest,
    wilcoxon_test,
    cohens_d,
    bootstrap_ci,
    pairwise_tests,
)


def compare_agents(
    agent_performances: Dict[str, np.ndarray],
    metrics: Optional[List[str]] = None,
    test_type: str = "ttest",
    correction_method: str = "holm",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Comprehensive pairwise comparison of agents across multiple metrics.
    
    Args:
        agent_performances: Dict mapping agent name to performance array
        metrics: List of metric names (if multiple metrics per agent)
        test_type: "ttest" or "wilcoxon"
        correction_method: Multiple comparison correction ("bonferroni", "holm", "fdr_bh")
        alpha: Significance level
        
    Returns:
        DataFrame with comparison results including:
            - agent1, agent2: agents being compared
            - mean_diff: mean performance difference
            - p_value: raw p-value
            - adjusted_p: corrected p-value
            - reject: whether null hypothesis is rejected
            - effect_size: Cohen's d
            - interpretation: effect size interpretation
            
    Example:
        >>> performances = {
        ...     "Momentum": np.array([0.01, 0.02, 0.03, ...]),
        ...     "Contrarian": np.array([0.005, 0.015, 0.01, ...]),
        ...     "Random": np.array([0.0, -0.01, 0.02, ...]),
        ... }
        >>> comparison = compare_agents(performances)
        >>> print(comparison[comparison['reject']])  # Significant differences
    """
    # Choose test function
    if test_type == "ttest":
        test_func = paired_ttest
    elif test_type == "wilcoxon":
        test_func = wilcoxon_test
    else:
        raise ValueError(f"Unknown test_type: {test_type}")
    
    # Perform pairwise tests
    results = pairwise_tests(
        agent_performances,
        test_func=test_func,
        correction_method=correction_method,
        alpha=alpha,
    )
    
    # Compute effect sizes
    effect_sizes = []
    for _, row in results.iterrows():
        agent1 = row["agent1"]
        agent2 = row["agent2"]
        d = cohens_d(agent_performances[agent1], agent_performances[agent2])
        effect_sizes.append(d)
    
    results["effect_size"] = effect_sizes
    
    # Interpret effect sizes
    def interpret_d(d: float) -> str:
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    results["interpretation"] = results["effect_size"].apply(interpret_d)
    
    return results


def rank_agents(
    agent_performances: Dict[str, np.ndarray],
    metric_name: str = "performance",
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
) -> pd.DataFrame:
    """Rank agents by mean performance with bootstrap confidence intervals.
    
    Args:
        agent_performances: Dict mapping agent name to performance array
        metric_name: Name of the performance metric (for display)
        confidence_level: Confidence level for bootstrap CI
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        DataFrame with columns:
            - rank: ordinal rank (1 = best)
            - agent: agent name
            - mean: mean performance
            - std: standard deviation
            - ci_lower: lower bound of CI
            - ci_upper: upper bound of CI
            - n_samples: number of observations
            
    Example:
        >>> rankings = rank_agents(performances, metric_name="Sharpe Ratio")
        >>> print(rankings)
    """
    results = []
    
    for agent_name, performance in agent_performances.items():
        mean_perf = np.mean(performance)
        std_perf = np.std(performance, ddof=1)
        
        # Bootstrap CI
        _, ci_lower, ci_upper = bootstrap_ci(
            performance,
            statistic_func=np.mean,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )
        
        results.append({
            "agent": agent_name,
            "mean": mean_perf,
            "std": std_perf,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_samples": len(performance),
        })
    
    # Sort by mean (descending)
    df = pd.DataFrame(results)
    df = df.sort_values("mean", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    
    return df


def dominance_analysis(
    agent_performances: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, List[str]]:
    """Identify which agents statistically dominate others.
    
    Agent A dominates agent B if A performs significantly better than B
    and never significantly worse.
    
    Args:
        agent_performances: Dict mapping agent name to performance array
        alpha: Significance level
        
    Returns:
        Dictionary mapping each agent to list of agents it dominates
        
    Example:
        >>> dominance = dominance_analysis(performances)
        >>> for agent, dominated in dominance.items():
        ...     if dominated:
        ...         print(f"{agent} dominates: {', '.join(dominated)}")
    """
    agent_names = list(agent_performances.keys())
    n_agents = len(agent_names)
    
    # Build dominance matrix
    dominates = {name: [] for name in agent_names}
    
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            
            agent_i = agent_names[i]
            agent_j = agent_names[j]
            
            # Test if agent_i > agent_j
            result = paired_ttest(
                agent_performances[agent_i],
                agent_performances[agent_j],
                alternative="greater",
            )
            
            if result["p_value"] < alpha:
                dominates[agent_i].append(agent_j)
    
    return dominates


def head_to_head(
    agent_performances: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute head-to-head win/loss/tie records for all agent pairs.
    
    For each pair of agents, count how many times each agent performed better.
    
    Args:
        agent_performances: Dict mapping agent name to performance array
        
    Returns:
        DataFrame with columns:
            - agent1, agent2
            - wins: times agent1 > agent2
            - losses: times agent1 < agent2
            - ties: times agent1 == agent2
            - win_rate: wins / (wins + losses + ties)
            
    Example:
        >>> h2h = head_to_head(performances)
        >>> print(h2h[h2h['agent1'] == 'Momentum'])
    """
    agent_names = list(agent_performances.keys())
    n_agents = len(agent_names)
    
    records = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            agent_i = agent_names[i]
            agent_j = agent_names[j]
            
            perf_i = agent_performances[agent_i]
            perf_j = agent_performances[agent_j]
            
            # Count wins/losses/ties
            wins = np.sum(perf_i > perf_j)
            losses = np.sum(perf_i < perf_j)
            ties = np.sum(perf_i == perf_j)
            total = wins + losses + ties
            
            win_rate = wins / total if total > 0 else 0.0
            
            records.append({
                "agent1": agent_i,
                "agent2": agent_j,
                "wins": int(wins),
                "losses": int(losses),
                "ties": int(ties),
                "win_rate": win_rate,
            })
    
    return pd.DataFrame(records)


def performance_consistency(
    agent_performances: Dict[str, np.ndarray],
    window_size: Optional[int] = None,
) -> pd.DataFrame:
    """Analyze performance consistency across time windows.
    
    Computes rolling mean and standard deviation to assess stability.
    
    Args:
        agent_performances: Dict mapping agent name to performance array
        window_size: Window size for rolling statistics (default: len/5)
        
    Returns:
        DataFrame with columns:
            - agent: agent name
            - overall_mean: mean over entire period
            - overall_std: standard deviation
            - rolling_std_mean: mean of rolling standard deviations
            - consistency_score: 1 / (1 + rolling_std_mean) [higher = more consistent]
            
    Example:
        >>> consistency = performance_consistency(performances)
        >>> print(consistency.sort_values('consistency_score', ascending=False))
    """
    results = []
    
    for agent_name, performance in agent_performances.items():
        overall_mean = np.mean(performance)
        overall_std = np.std(performance, ddof=1)
        
        # Rolling statistics
        if window_size is None:
            window_size = max(5, len(performance) // 5)
        
        if len(performance) >= window_size:
            rolling_means = []
            rolling_stds = []
            for i in range(len(performance) - window_size + 1):
                window = performance[i:i + window_size]
                rolling_means.append(np.mean(window))
                rolling_stds.append(np.std(window, ddof=1))
            
            rolling_std_mean = np.mean(rolling_stds)
            consistency_score = 1.0 / (1.0 + rolling_std_mean)
        else:
            rolling_std_mean = overall_std
            consistency_score = 1.0 / (1.0 + overall_std)
        
        results.append({
            "agent": agent_name,
            "overall_mean": overall_mean,
            "overall_std": overall_std,
            "rolling_std_mean": rolling_std_mean,
            "consistency_score": consistency_score,
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("consistency_score", ascending=False).reset_index(drop=True)
    
    return df


def pareto_frontier(
    agent_performances: Dict[str, Dict[str, float]],
    objectives: List[str],
    maximize: Optional[List[bool]] = None,
) -> Tuple[List[str], List[str]]:
    """Identify Pareto-optimal agents given multiple objectives.
    
    An agent is Pareto-optimal if no other agent is better in all objectives.
    
    Args:
        agent_performances: Dict mapping agent to dict of objective values
            Example: {"Momentum": {"return": 0.05, "risk": 0.02}, ...}
        objectives: List of objective names to consider
        maximize: List of bools indicating whether to maximize each objective
            (default: all True)
        
    Returns:
        Tuple of (pareto_optimal_agents, dominated_agents)
        
    Example:
        >>> perfs = {
        ...     "A": {"return": 0.10, "sharpe": 1.5},
        ...     "B": {"return": 0.08, "sharpe": 2.0},
        ...     "C": {"return": 0.05, "sharpe": 1.0},
        ... }
        >>> optimal, dominated = pareto_frontier(perfs, ["return", "sharpe"])
        >>> print(f"Pareto optimal: {optimal}")
    """
    agent_names = list(agent_performances.keys())
    n_agents = len(agent_names)
    
    if maximize is None:
        maximize = [True] * len(objectives)
    
    if len(maximize) != len(objectives):
        raise ValueError("maximize must have same length as objectives")
    
    # Build objective matrix
    obj_matrix = np.zeros((n_agents, len(objectives)))
    for i, agent in enumerate(agent_names):
        for j, obj in enumerate(objectives):
            value = agent_performances[agent][obj]
            # Negate if minimizing
            obj_matrix[i, j] = value if maximize[j] else -value
    
    # Find Pareto-optimal points
    is_pareto = np.ones(n_agents, dtype=bool)
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            # Check if j dominates i
            if np.all(obj_matrix[j] >= obj_matrix[i]) and np.any(obj_matrix[j] > obj_matrix[i]):
                is_pareto[i] = False
                break
    
    pareto_optimal = [agent_names[i] for i in range(n_agents) if is_pareto[i]]
    dominated = [agent_names[i] for i in range(n_agents) if not is_pareto[i]]
    
    return pareto_optimal, dominated


def create_comparison_matrix(
    agent_performances: Dict[str, np.ndarray],
    test_type: str = "ttest",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Create a matrix showing pairwise comparison results.
    
    Args:
        agent_performances: Dict mapping agent name to performance array
        test_type: "ttest" or "wilcoxon"
        alpha: Significance level
        
    Returns:
        DataFrame where cell [i,j] indicates relationship between agent i and j:
            - ">>" : agent i significantly better than j
            - "<<" : agent i significantly worse than j
            - "≈" : no significant difference
            
    Example:
        >>> matrix = create_comparison_matrix(performances)
        >>> print(matrix)
    """
    agent_names = list(agent_performances.keys())
    n_agents = len(agent_names)
    
    # Initialize matrix
    matrix = np.full((n_agents, n_agents), "≈", dtype=object)
    
    # Fill diagonal
    for i in range(n_agents):
        matrix[i, i] = "="
    
    # Test function
    if test_type == "ttest":
        test_func = paired_ttest
    elif test_type == "wilcoxon":
        test_func = wilcoxon_test
    else:
        raise ValueError(f"Unknown test_type: {test_type}")
    
    # Pairwise comparisons
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            agent_i = agent_names[i]
            agent_j = agent_names[j]
            
            result = test_func(
                agent_performances[agent_i],
                agent_performances[agent_j],
                alternative="two-sided",
            )
            
            if result["p_value"] < alpha:
                mean_i = np.mean(agent_performances[agent_i])
                mean_j = np.mean(agent_performances[agent_j])
                
                if mean_i > mean_j:
                    matrix[i, j] = ">>"
                    matrix[j, i] = "<<"
                else:
                    matrix[i, j] = "<<"
                    matrix[j, i] = ">>"
    
    # Create DataFrame
    df = pd.DataFrame(matrix, index=agent_names, columns=agent_names)
    
    return df


def summary_report(
    agent_performances: Dict[str, np.ndarray],
    metric_name: str = "Return",
) -> str:
    """Generate a comprehensive text summary of agent comparisons.
    
    Args:
        agent_performances: Dict mapping agent name to performance array
        metric_name: Name of the metric being compared
        
    Returns:
        Formatted string report
        
    Example:
        >>> report = summary_report(performances, metric_name="Sharpe Ratio")
        >>> print(report)
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"AGENT COMPARISON SUMMARY: {metric_name}")
    lines.append("=" * 70)
    lines.append("")
    
    # Rankings
    rankings = rank_agents(agent_performances, metric_name)
    lines.append("### Rankings ###")
    lines.append("")
    for _, row in rankings.iterrows():
        lines.append(
            f"{row['rank']}. {row['agent']}: "
            f"{row['mean']:.4f} ± {row['std']:.4f} "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        )
    lines.append("")
    
    # Pairwise comparisons
    comparison = compare_agents(agent_performances)
    significant = comparison[comparison["reject"]]
    
    lines.append("### Significant Differences ###")
    lines.append("")
    if len(significant) == 0:
        lines.append("No significant differences found.")
    else:
        for _, row in significant.iterrows():
            symbol = ">" if row["mean_diff"] > 0 else "<"
            lines.append(
                f"{row['agent1']} {symbol} {row['agent2']}: "
                f"p={row['adjusted_p']:.4f}, d={row['effect_size']:.3f} ({row['interpretation']})"
            )
    lines.append("")
    
    # Dominance
    dominance = dominance_analysis(agent_performances)
    lines.append("### Dominance Analysis ###")
    lines.append("")
    has_dominance = False
    for agent, dominated_list in dominance.items():
        if dominated_list:
            has_dominance = True
            lines.append(f"{agent} dominates: {', '.join(dominated_list)}")
    if not has_dominance:
        lines.append("No clear dominance detected.")
    lines.append("")
    
    # Consistency
    consistency = performance_consistency(agent_performances)
    lines.append("### Consistency Scores ###")
    lines.append("")
    for _, row in consistency.iterrows():
        lines.append(
            f"{row['agent']}: {row['consistency_score']:.4f} "
            f"(rolling_std={row['rolling_std_mean']:.4f})"
        )
    lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)

