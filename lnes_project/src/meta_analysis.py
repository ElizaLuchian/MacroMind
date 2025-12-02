"""Meta-analysis tools for aggregating results across multiple studies.

This module provides methods for combining results from multiple experimental
runs, computing effect sizes, and detecting publication bias.

Functions:
    aggregate_results: Combine results from multiple runs
    forest_plot: Visualize effect sizes with confidence intervals
    heterogeneity_test: Test for heterogeneity across studies (I² statistic)
    funnel_plot: Visualize publication bias
    meta_regression: Explain heterogeneity using moderator variables
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats


def aggregate_results(
    results: List[Dict[str, float]],
    method: str = "fixed",
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Aggregate effect sizes from multiple studies using meta-analysis.
    
    Args:
        results: List of dicts with keys: 'effect_size', 'variance', 'n'
        method: "fixed" (fixed-effect) or "random" (random-effects)
        confidence_level: Confidence level for CI
        
    Returns:
        Dictionary with:
            - pooled_effect: combined effect size
            - ci_lower, ci_upper: confidence interval
            - p_value: p-value for pooled effect
            - heterogeneity: I² statistic
            - tau_squared: between-study variance (random-effects only)
            
    Example:
        >>> studies = [
        ...     {'effect_size': 0.5, 'variance': 0.01, 'n': 100},
        ...     {'effect_size': 0.6, 'variance': 0.02, 'n': 80},
        ...     {'effect_size': 0.4, 'variance': 0.015, 'n': 120},
        ... ]
        >>> meta = aggregate_results(studies, method="random")
        >>> print(f"Pooled effect: {meta['pooled_effect']:.3f} "
        ...       f"[{meta['ci_lower']:.3f}, {meta['ci_upper']:.3f}]")
    """
    if len(results) == 0:
        return {
            "pooled_effect": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "heterogeneity": np.nan,
            "tau_squared": np.nan,
        }
    
    # Extract data
    effects = np.array([r['effect_size'] for r in results])
    variances = np.array([r['variance'] for r in results])
    weights_fe = 1.0 / variances  # Fixed-effect weights
    
    # Fixed-effect pooled estimate
    pooled_fe = np.sum(weights_fe * effects) / np.sum(weights_fe)
    var_pooled_fe = 1.0 / np.sum(weights_fe)
    
    # Heterogeneity test (Q statistic)
    Q = np.sum(weights_fe * (effects - pooled_fe) ** 2)
    df = len(results) - 1
    p_het = 1.0 - stats.chi2.cdf(Q, df) if df > 0 else 1.0
    
    # I² statistic
    I_squared = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    
    # Random-effects model
    if method == "random":
        # DerSimonian-Laird estimator for tau²
        if df > 0:
            C = np.sum(weights_fe) - np.sum(weights_fe ** 2) / np.sum(weights_fe)
            tau_squared = max(0.0, (Q - df) / C)
        else:
            tau_squared = 0.0
        
        # Random-effects weights
        weights_re = 1.0 / (variances + tau_squared)
        pooled_effect = np.sum(weights_re * effects) / np.sum(weights_re)
        var_pooled = 1.0 / np.sum(weights_re)
    else:
        pooled_effect = pooled_fe
        var_pooled = var_pooled_fe
        tau_squared = 0.0
    
    # Confidence interval
    z_alpha = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    se_pooled = np.sqrt(var_pooled)
    ci_lower = pooled_effect - z_alpha * se_pooled
    ci_upper = pooled_effect + z_alpha * se_pooled
    
    # P-value
    z_stat = pooled_effect / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        "pooled_effect": float(pooled_effect),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "se": float(se_pooled),
        "p_value": float(p_value),
        "heterogeneity": float(I_squared),
        "Q_statistic": float(Q),
        "Q_p_value": float(p_het),
        "tau_squared": float(tau_squared),
        "n_studies": len(results),
    }


def compute_heterogeneity(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
) -> Dict[str, float]:
    """Compute heterogeneity statistics for a set of studies.
    
    Args:
        effect_sizes: Array of effect sizes
        variances: Array of variances
        
    Returns:
        Dictionary with heterogeneity metrics:
            - Q: Cochran's Q statistic
            - Q_p_value: p-value for Q test
            - I_squared: I² statistic (% of variability due to heterogeneity)
            - tau_squared: Between-study variance
            
    Example:
        >>> het = compute_heterogeneity(effects, variances)
        >>> if het['I_squared'] > 75:
        ...     print("High heterogeneity detected")
    """
    weights = 1.0 / variances
    pooled = np.sum(weights * effect_sizes) / np.sum(weights)
    
    Q = np.sum(weights * (effect_sizes - pooled) ** 2)
    df = len(effect_sizes) - 1
    Q_p_value = 1.0 - stats.chi2.cdf(Q, df) if df > 0 else 1.0
    
    I_squared = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    
    # Tau²
    if df > 0:
        C = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
        tau_squared = max(0.0, (Q - df) / C)
    else:
        tau_squared = 0.0
    
    return {
        "Q": float(Q),
        "Q_p_value": float(Q_p_value),
        "I_squared": float(I_squared),
        "tau_squared": float(tau_squared),
    }


def subgroup_analysis(
    results: pd.DataFrame,
    subgroup_column: str,
    effect_column: str = "effect_size",
    variance_column: str = "variance",
) -> pd.DataFrame:
    """Perform subgroup meta-analysis.
    
    Args:
        results: DataFrame with study results
        subgroup_column: Column defining subgroups
        effect_column: Column with effect sizes
        variance_column: Column with variances
        
    Returns:
        DataFrame with meta-analysis results for each subgroup
        
    Example:
        >>> results_df = pd.DataFrame({
        ...     'study': ['A', 'B', 'C', 'D'],
        ...     'effect_size': [0.5, 0.6, 0.3, 0.4],
        ...     'variance': [0.01, 0.02, 0.01, 0.015],
        ...     'dataset': ['small', 'small', 'large', 'large'],
        ...     'n': [100, 80, 200, 180],
        ... })
        >>> subgroup = subgroup_analysis(results_df, 'dataset')
        >>> print(subgroup)
    """
    subgroups = results[subgroup_column].unique()
    subgroup_results = []
    
    for subgroup in subgroups:
        subgroup_data = results[results[subgroup_column] == subgroup]
        
        studies = []
        for _, row in subgroup_data.iterrows():
            studies.append({
                'effect_size': row[effect_column],
                'variance': row[variance_column],
                'n': row.get('n', 100),
            })
        
        meta_result = aggregate_results(studies, method="random")
        meta_result['subgroup'] = subgroup
        meta_result['n_studies'] = len(studies)
        
        subgroup_results.append(meta_result)
    
    return pd.DataFrame(subgroup_results)


def cumulative_meta_analysis(
    results: List[Dict[str, float]],
    sort_by: str = "year",
) -> pd.DataFrame:
    """Perform cumulative meta-analysis (adding one study at a time).
    
    Args:
        results: List of study results
        sort_by: Key to sort studies by (e.g., 'year', 'n')
        
    Returns:
        DataFrame showing cumulative pooled effect as studies are added
        
    Example:
        >>> cumulative = cumulative_meta_analysis(studies, sort_by='year')
        >>> # Plot how pooled effect stabilizes over time
        >>> plt.plot(cumulative['n_studies'], cumulative['pooled_effect'])
    """
    if sort_by in results[0]:
        results_sorted = sorted(results, key=lambda x: x[sort_by])
    else:
        results_sorted = results
    
    cumulative_results = []
    
    for i in range(1, len(results_sorted) + 1):
        subset = results_sorted[:i]
        meta = aggregate_results(subset, method="random")
        meta['n_studies'] = i
        cumulative_results.append(meta)
    
    return pd.DataFrame(cumulative_results)


def publication_bias_test(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    method: str = "egger",
) -> Dict[str, float]:
    """Test for publication bias.
    
    Args:
        effect_sizes: Array of effect sizes
        variances: Array of variances
        method: "egger" (Egger's test) or "begg" (Begg's test)
        
    Returns:
        Dictionary with test results:
            - test_statistic: test statistic value
            - p_value: p-value
            - interpretation: text interpretation
            
    Example:
        >>> bias_test = publication_bias_test(effects, variances)
        >>> if bias_test['p_value'] < 0.05:
        ...     print("Potential publication bias detected")
    """
    n = len(effect_sizes)
    
    if method == "egger":
        # Egger's regression test
        # Regress standardized effect on precision
        se = np.sqrt(variances)
        precision = 1.0 / se
        standardized_effect = effect_sizes / se
        
        # Linear regression: standardized_effect ~ precision
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(precision, standardized_effect)
        
        # Test if intercept significantly different from 0
        t_stat = intercept / std_err if std_err > 0 else 0.0
        df = n - 2
        p_value_bias = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        interpretation = "No evidence of bias" if p_value_bias > 0.05 else "Potential bias detected"
        
        return {
            "test_statistic": float(t_stat),
            "p_value": float(p_value_bias),
            "intercept": float(intercept),
            "interpretation": interpretation,
        }
    
    elif method == "begg":
        # Begg's rank correlation test
        # Rank correlation between effect sizes and variances
        from scipy.stats import kendalltau
        tau, p_value = kendalltau(effect_sizes, variances)
        
        interpretation = "No evidence of bias" if p_value > 0.05 else "Potential bias detected"
        
        return {
            "test_statistic": float(tau),
            "p_value": float(p_value),
            "interpretation": interpretation,
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def forest_plot_data(
    results: List[Dict[str, Any]],
    study_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Prepare data for forest plot visualization.
    
    Args:
        results: List of study results with effect_size, variance, ci_lower, ci_upper
        study_labels: Optional labels for studies
        
    Returns:
        DataFrame formatted for forest plot
        
    Example:
        >>> plot_data = forest_plot_data(studies, labels=['Study A', 'Study B', ...])
        >>> # Use with matplotlib to create forest plot
    """
    if study_labels is None:
        study_labels = [f"Study {i+1}" for i in range(len(results))]
    
    plot_data = []
    for i, (result, label) in enumerate(zip(results, study_labels)):
        effect = result['effect_size']
        se = np.sqrt(result['variance'])
        ci_lower = result.get('ci_lower', effect - 1.96 * se)
        ci_upper = result.get('ci_upper', effect + 1.96 * se)
        
        plot_data.append({
            'study': label,
            'effect_size': effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'weight': result.get('n', 100),
        })
    
    # Add pooled estimate
    meta_result = aggregate_results(results, method="random")
    plot_data.append({
        'study': 'Pooled (Random)',
        'effect_size': meta_result['pooled_effect'],
        'ci_lower': meta_result['ci_lower'],
        'ci_upper': meta_result['ci_upper'],
        'weight': sum(r.get('n', 100) for r in results),
    })
    
    return pd.DataFrame(plot_data)


def visualize_forest_plot(
    plot_data: pd.DataFrame,
    title: str = "Forest Plot",
    xlabel: str = "Effect Size",
) -> None:
    """Create a forest plot visualization.
    
    Args:
        plot_data: DataFrame from forest_plot_data()
        title: Plot title
        xlabel: X-axis label
        
    Example:
        >>> visualize_forest_plot(plot_data)
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    
    n_studies = len(plot_data)
    y_positions = np.arange(n_studies)
    
    fig, ax = plt.subplots(figsize=(10, max(6, n_studies * 0.5)))
    
    # Plot confidence intervals
    for i, row in plot_data.iterrows():
        y = y_positions[i]
        # CI line
        ax.plot([row['ci_lower'], row['ci_upper']], [y, y], 'k-', linewidth=1.5)
        # Point estimate
        if i == n_studies - 1:  # Pooled
            ax.plot(row['effect_size'], y, 'D', markersize=10, color='red', label='Pooled')
        else:
            size = np.sqrt(row['weight']) / 2
            ax.plot(row['effect_size'], y, 'o', markersize=max(4, size), color='blue')
    
    # Null line
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data['study'])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    fig.tight_layout()


def visualize_funnel_plot(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    title: str = "Funnel Plot",
) -> None:
    """Create a funnel plot for publication bias assessment.
    
    Args:
        effect_sizes: Array of effect sizes
        variances: Array of variances
        title: Plot title
        
    Example:
        >>> visualize_funnel_plot(effects, variances)
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    
    se = np.sqrt(variances)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(effect_sizes, se, s=50, alpha=0.6)
    
    # Funnel (expected range under no bias)
    pooled = np.average(effect_sizes, weights=1.0/variances)
    max_se = np.max(se)
    funnel_x = [pooled - 1.96*max_se, pooled, pooled + 1.96*max_se]
    funnel_y = [max_se, 0, max_se]
    ax.plot(funnel_x, funnel_y, 'k--', alpha=0.5, label='95% CI')
    
    ax.axvline(pooled, color='red', linestyle='--', linewidth=2, label='Pooled Effect')
    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Standard Error')
    ax.set_title(title)
    ax.invert_yaxis()  # Larger studies at top
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()


def generate_meta_analysis_report(
    results: List[Dict[str, float]],
    study_labels: Optional[List[str]] = None,
) -> str:
    """Generate comprehensive meta-analysis report.
    
    Args:
        results: List of study results
        study_labels: Optional study labels
        
    Returns:
        Formatted text report
        
    Example:
        >>> report = generate_meta_analysis_report(studies)
        >>> print(report)
    """
    lines = []
    lines.append("=" * 70)
    lines.append("META-ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Summary
    lines.append(f"Number of studies: {len(results)}")
    lines.append("")
    
    # Fixed-effect meta-analysis
    meta_fe = aggregate_results(results, method="fixed")
    lines.append("### Fixed-Effect Model ###")
    lines.append(f"Pooled effect: {meta_fe['pooled_effect']:.4f} "
                f"[{meta_fe['ci_lower']:.4f}, {meta_fe['ci_upper']:.4f}]")
    lines.append(f"p-value: {meta_fe['p_value']:.4f}")
    lines.append("")
    
    # Random-effects meta-analysis
    meta_re = aggregate_results(results, method="random")
    lines.append("### Random-Effects Model ###")
    lines.append(f"Pooled effect: {meta_re['pooled_effect']:.4f} "
                f"[{meta_re['ci_lower']:.4f}, {meta_re['ci_upper']:.4f}]")
    lines.append(f"p-value: {meta_re['p_value']:.4f}")
    lines.append(f"τ² (tau-squared): {meta_re['tau_squared']:.4f}")
    lines.append("")
    
    # Heterogeneity
    lines.append("### Heterogeneity ###")
    lines.append(f"Q statistic: {meta_re['Q_statistic']:.4f} (p={meta_re['Q_p_value']:.4f})")
    lines.append(f"I² statistic: {meta_re['heterogeneity']:.1f}%")
    
    if meta_re['heterogeneity'] < 25:
        het_interpretation = "Low heterogeneity"
    elif meta_re['heterogeneity'] < 50:
        het_interpretation = "Moderate heterogeneity"
    elif meta_re['heterogeneity'] < 75:
        het_interpretation = "Substantial heterogeneity"
    else:
        het_interpretation = "High heterogeneity"
    
    lines.append(f"Interpretation: {het_interpretation}")
    lines.append("")
    
    # Publication bias
    effects = np.array([r['effect_size'] for r in results])
    variances = np.array([r['variance'] for r in results])
    
    if len(results) >= 3:
        bias_test = publication_bias_test(effects, variances, method="egger")
        lines.append("### Publication Bias (Egger's Test) ###")
        lines.append(f"Test statistic: {bias_test['test_statistic']:.4f}")
        lines.append(f"p-value: {bias_test['p_value']:.4f}")
        lines.append(f"Interpretation: {bias_test['interpretation']}")
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)

