"""Statistical testing utilities for agent comparison and hypothesis testing.

This module provides statistical tools for rigorous comparison of trading agents,
including parametric and non-parametric tests, multiple comparison corrections,
effect size calculations, and bootstrap confidence intervals.

Functions:
    paired_ttest: Paired t-test for comparing two agents
    wilcoxon_test: Non-parametric alternative to paired t-test
    mcnemar_test: Test for binary decision differences
    cohens_d: Effect size calculation
    bootstrap_ci: Bootstrap confidence intervals
    permutation_test: Permutation-based significance test
    multiple_comparison_correction: Bonferroni, Holm, FDR corrections
    
References:
    - Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets
    - Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised
      classification learning algorithms
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Literal
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as t_dist


def paired_ttest(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> Dict[str, float]:
    """Perform paired t-test comparing two samples.
    
    Use when comparing the same agents on the same data (paired observations).
    Assumes normally distributed differences.
    
    Args:
        sample1: First sample (e.g., agent A returns on each day)
        sample2: Second sample (e.g., agent B returns on same days)
        alternative: Type of test - "two-sided", "greater" (sample1 > sample2), or "less"
        
    Returns:
        Dictionary with keys:
            - t_statistic: t-statistic value
            - p_value: p-value for the test
            - dof: degrees of freedom
            - mean_diff: mean difference (sample1 - sample2)
            - stderr: standard error of the difference
            - ci_lower: lower bound of 95% CI for mean difference
            - ci_upper: upper bound of 95% CI for mean difference
            
    Example:
        >>> agent_a_returns = np.array([0.01, 0.02, -0.01, 0.03])
        >>> agent_b_returns = np.array([0.005, 0.015, -0.005, 0.02])
        >>> result = paired_ttest(agent_a_returns, agent_b_returns)
        >>> print(f"p-value: {result['p_value']:.4f}")
    """
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    
    if len(sample1) != len(sample2):
        raise ValueError(f"Samples must have equal length: {len(sample1)} != {len(sample2)}")
    
    if len(sample1) < 2:
        raise ValueError("Need at least 2 paired observations")
    
    # Compute differences
    diff = sample1 - sample2
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    stderr = std_diff / np.sqrt(n)
    
    # t-statistic
    t_stat = mean_diff / stderr if stderr > 0 else 0.0
    dof = n - 1
    
    # p-value
    if alternative == "two-sided":
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), dof))
    elif alternative == "greater":
        p_value = 1 - t_dist.cdf(t_stat, dof)
    elif alternative == "less":
        p_value = t_dist.cdf(t_stat, dof)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")
    
    # 95% confidence interval
    ci_delta = t_dist.ppf(0.975, dof) * stderr
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "dof": int(dof),
        "mean_diff": float(mean_diff),
        "stderr": float(stderr),
        "ci_lower": float(mean_diff - ci_delta),
        "ci_upper": float(mean_diff + ci_delta),
    }


def wilcoxon_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> Dict[str, float]:
    """Perform Wilcoxon signed-rank test (non-parametric paired test).
    
    Use when differences are not normally distributed or sample size is small.
    Tests whether median difference is zero.
    
    Args:
        sample1: First sample
        sample2: Second sample (paired with sample1)
        alternative: "two-sided", "greater", or "less"
        
    Returns:
        Dictionary with keys:
            - statistic: Wilcoxon test statistic
            - p_value: p-value
            - median_diff: median of differences
            
    Example:
        >>> result = wilcoxon_test(returns_a, returns_b)
        >>> if result['p_value'] < 0.05:
        ...     print("Significant difference detected")
    """
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    
    if len(sample1) != len(sample2):
        raise ValueError("Samples must have equal length")
    
    diff = sample1 - sample2
    
    # Remove zeros
    diff_nonzero = diff[diff != 0]
    
    if len(diff_nonzero) == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "median_diff": 0.0,
        }
    
    result = stats.wilcoxon(diff_nonzero, alternative=alternative)
    
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "median_diff": float(np.median(diff)),
    }


def mcnemar_test(
    agent1_correct: np.ndarray,
    agent2_correct: np.ndarray,
) -> Dict[str, float]:
    """Perform McNemar's test for comparing binary decision correctness.
    
    Tests whether two agents have significantly different error rates on the
    same test set. Useful for comparing directional accuracy.
    
    Args:
        agent1_correct: Boolean array indicating correctness for agent 1
        agent2_correct: Boolean array indicating correctness for agent 2
        
    Returns:
        Dictionary with keys:
            - statistic: McNemar test statistic
            - p_value: p-value
            - contingency_table: 2x2 contingency table as dict
            
    Example:
        >>> correct_a = np.array([True, True, False, True, False])
        >>> correct_b = np.array([True, False, False, True, True])
        >>> result = mcnemar_test(correct_a, correct_b)
        >>> print(f"Different error rates? p={result['p_value']:.4f}")
    """
    agent1_correct = np.asarray(agent1_correct, dtype=bool)
    agent2_correct = np.asarray(agent2_correct, dtype=bool)
    
    if len(agent1_correct) != len(agent2_correct):
        raise ValueError("Arrays must have equal length")
    
    # Build contingency table
    both_correct = np.sum(agent1_correct & agent2_correct)
    both_wrong = np.sum(~agent1_correct & ~agent2_correct)
    a1_correct_a2_wrong = np.sum(agent1_correct & ~agent2_correct)
    a1_wrong_a2_correct = np.sum(~agent1_correct & agent2_correct)
    
    # McNemar's test uses the discordant pairs
    b = a1_correct_a2_wrong
    c = a1_wrong_a2_correct
    
    if b + c == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "contingency_table": {
                "both_correct": int(both_correct),
                "both_wrong": int(both_wrong),
                "a1_correct_a2_wrong": int(b),
                "a1_wrong_a2_correct": int(c),
            },
        }
    
    # Test statistic with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "contingency_table": {
            "both_correct": int(both_correct),
            "both_wrong": int(both_wrong),
            "a1_correct_a2_wrong": int(b),
            "a1_wrong_a2_correct": int(c),
        },
    }


def cohens_d(
    sample1: np.ndarray,
    sample2: np.ndarray,
    pooled: bool = True,
) -> float:
    """Calculate Cohen's d effect size for two samples.
    
    Effect size interpretation (Cohen's guidelines):
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large
        
    Args:
        sample1: First sample
        sample2: Second sample
        pooled: If True, use pooled standard deviation; if False, use sample1's std
        
    Returns:
        Cohen's d effect size
        
    Example:
        >>> d = cohens_d(momentum_returns, random_returns)
        >>> print(f"Effect size: {d:.3f} ({'large' if abs(d) >= 0.8 else 'small'})")
    """
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    
    mean_diff = np.mean(sample1) - np.mean(sample2)
    
    if pooled:
        n1, n2 = len(sample1), len(sample2)
        var1 = np.var(sample1, ddof=1)
        var2 = np.var(sample2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return mean_diff / pooled_std if pooled_std > 0 else 0.0
    else:
        std1 = np.std(sample1, ddof=1)
        return mean_diff / std1 if std1 > 0 else 0.0


def bootstrap_ci(
    sample: np.ndarray,
    statistic_func=np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.
    
    Args:
        sample: Data sample
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (statistic_value, ci_lower, ci_upper)
        
    Example:
        >>> sharpe = lambda x: np.mean(x) / np.std(x) if np.std(x) > 0 else 0
        >>> stat, lower, upper = bootstrap_ci(returns, sharpe, n_bootstrap=5000)
        >>> print(f"Sharpe: {stat:.3f} [{lower:.3f}, {upper:.3f}]")
    """
    sample = np.asarray(sample)
    n = len(sample)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Compute original statistic
    original_stat = statistic_func(sample)
    
    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample = np.random.choice(sample, size=n, replace=True)
        bootstrap_stats[i] = statistic_func(resample)
    
    # Compute percentile confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return float(original_stat), float(ci_lower), float(ci_upper)


def permutation_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    statistic_func=lambda x, y: np.mean(x) - np.mean(y),
    n_permutations: int = 10000,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    random_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Perform permutation test for comparing two samples.
    
    Non-parametric test that makes no distributional assumptions.
    
    Args:
        sample1: First sample
        sample2: Second sample
        statistic_func: Function computing test statistic from two samples
        n_permutations: Number of random permutations
        alternative: "two-sided", "greater", or "less"
        random_seed: Random seed
        
    Returns:
        Dictionary with keys:
            - observed_statistic: observed test statistic
            - p_value: permutation p-value
            - n_permutations: number of permutations performed
            
    Example:
        >>> result = permutation_test(agent_a_pnl, agent_b_pnl, n_permutations=5000)
        >>> print(f"Permutation p-value: {result['p_value']:.4f}")
    """
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Observed statistic
    observed = statistic_func(sample1, sample2)
    
    # Combine samples
    combined = np.concatenate([sample1, sample2])
    n1 = len(sample1)
    n_total = len(combined)
    
    # Permutation distribution
    perm_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_combined = np.random.permutation(combined)
        perm_sample1 = perm_combined[:n1]
        perm_sample2 = perm_combined[n1:]
        perm_stats[i] = statistic_func(perm_sample1, perm_sample2)
    
    # Compute p-value
    if alternative == "two-sided":
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
    elif alternative == "greater":
        p_value = np.mean(perm_stats >= observed)
    elif alternative == "less":
        p_value = np.mean(perm_stats <= observed)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")
    
    return {
        "observed_statistic": float(observed),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
    }


def multiple_comparison_correction(
    p_values: List[float],
    method: Literal["bonferroni", "holm", "fdr_bh"] = "holm",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply multiple comparison correction to p-values.
    
    Methods:
        - bonferroni: Bonferroni correction (most conservative)
        - holm: Holm-Bonferroni (sequential Bonferroni)
        - fdr_bh: Benjamini-Hochberg FDR control (least conservative)
        
    Args:
        p_values: List of p-values from multiple tests
        method: Correction method
        alpha: Family-wise error rate (for bonferroni/holm) or FDR level (for fdr_bh)
        
    Returns:
        DataFrame with columns:
            - original_p: original p-value
            - adjusted_p: adjusted p-value
            - reject: whether to reject null hypothesis at alpha level
            - rank: rank of p-value (for sequential methods)
            
    Example:
        >>> p_vals = [0.01, 0.04, 0.03, 0.50, 0.008]
        >>> result = multiple_comparison_correction(p_vals, method="holm")
        >>> print(result)
    """
    p_values = np.asarray(p_values)
    n = len(p_values)
    
    if n == 0:
        return pd.DataFrame(columns=["original_p", "adjusted_p", "reject", "rank"])
    
    # Sort indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Compute adjusted p-values
    if method == "bonferroni":
        adjusted_p = np.minimum(sorted_p * n, 1.0)
    
    elif method == "holm":
        adjusted_p = np.zeros(n)
        for i in range(n):
            adjusted_p[i] = min((n - i) * sorted_p[i], 1.0)
        # Enforce monotonicity
        for i in range(1, n):
            adjusted_p[i] = max(adjusted_p[i], adjusted_p[i - 1])
    
    elif method == "fdr_bh":
        adjusted_p = np.zeros(n)
        for i in range(n):
            adjusted_p[i] = min(n / (i + 1) * sorted_p[i], 1.0)
        # Enforce monotonicity (reverse)
        for i in range(n - 2, -1, -1):
            adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Unsort
    unsorted_adjusted = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        unsorted_adjusted[idx] = adjusted_p[i]
    
    # Create result DataFrame
    result = pd.DataFrame({
        "original_p": p_values,
        "adjusted_p": unsorted_adjusted,
        "reject": unsorted_adjusted < alpha,
        "rank": [int(np.where(sorted_indices == i)[0][0] + 1) for i in range(n)],
    })
    
    return result


def pairwise_tests(
    samples: Dict[str, np.ndarray],
    test_func=paired_ttest,
    correction_method: Optional[str] = "holm",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Perform all pairwise statistical tests with multiple comparison correction.
    
    Args:
        samples: Dictionary mapping agent names to their performance samples
        test_func: Statistical test function (default: paired_ttest)
        correction_method: Multiple comparison correction method (None to skip)
        alpha: Significance level
        
    Returns:
        DataFrame with pairwise comparison results
        
    Example:
        >>> samples = {
        ...     "Momentum": momentum_returns,
        ...     "Contrarian": contrarian_returns,
        ...     "Random": random_returns,
        ... }
        >>> result = pairwise_tests(samples)
        >>> print(result[result['reject']])  # Show significant differences
    """
    agent_names = list(samples.keys())
    n_agents = len(agent_names)
    
    # Perform all pairwise tests
    comparisons = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            agent1 = agent_names[i]
            agent2 = agent_names[j]
            
            sample1 = samples[agent1]
            sample2 = samples[agent2]
            
            # Run test
            test_result = test_func(sample1, sample2)
            
            # Extract p-value
            p_value = test_result.get("p_value", np.nan)
            
            comparisons.append({
                "agent1": agent1,
                "agent2": agent2,
                "p_value": p_value,
                "test_statistic": test_result.get("t_statistic", test_result.get("statistic", np.nan)),
                "mean_diff": test_result.get("mean_diff", np.nan),
            })
    
    result_df = pd.DataFrame(comparisons)
    
    # Apply multiple comparison correction
    if correction_method is not None and len(comparisons) > 0:
        correction = multiple_comparison_correction(
            result_df["p_value"].values,
            method=correction_method,
            alpha=alpha,
        )
        result_df["adjusted_p"] = correction["adjusted_p"].values
        result_df["reject"] = correction["reject"].values
    else:
        result_df["adjusted_p"] = result_df["p_value"]
        result_df["reject"] = result_df["p_value"] < alpha
    
    return result_df

