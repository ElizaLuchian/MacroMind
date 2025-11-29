"""Ablation study framework for analyzing component importance.

This module provides tools for systematic ablation studies to understand
which components of the system contribute most to performance.

Functions:
    ablation_study: Remove one component at a time and measure impact
    component_importance: Rank components by their contribution
    interaction_effects: Analyze interactions between components
    full_factorial: Full factorial design for complete analysis
"""

from __future__ import annotations

from typing import Dict, List, Callable, Any, Optional, Tuple
from itertools import combinations
import warnings

import numpy as np
import pandas as pd


def ablation_study(
    baseline_func: Callable,
    components: Dict[str, Any],
    performance_metric: Callable[[Any], float],
    baseline_performance: Optional[float] = None,
) -> pd.DataFrame:
    """Perform ablation study by removing one component at a time.
    
    Args:
        baseline_func: Function that runs experiment with given components
            Signature: baseline_func(components_dict) -> result
        components: Dict mapping component names to their values
        performance_metric: Function extracting performance from result
            Signature: performance_metric(result) -> float
        baseline_performance: Pre-computed baseline (full system) performance
            If None, computed automatically
            
    Returns:
        DataFrame with columns:
            - component: component removed
            - performance: performance without this component
            - baseline_performance: performance with all components
            - delta: performance - baseline_performance
            - delta_pct: percentage change
            - importance: |delta| (absolute importance)
            
    Example:
        >>> def run_experiment(components):
        ...     # Run trading simulation with given components
        ...     return simulation_result
        >>> 
        >>> def get_sharpe(result):
        ...     return result['sharpe_ratio']
        >>> 
        >>> components = {
        ...     'news_clustering': True,
        ...     'sentiment_analysis': True,
        ...     'momentum_signal': True,
        ... }
        >>> 
        >>> results = ablation_study(run_experiment, components, get_sharpe)
        >>> print(results.sort_values('importance', ascending=False))
    """
    # Compute baseline performance (all components present)
    if baseline_performance is None:
        baseline_result = baseline_func(components)
        baseline_performance = performance_metric(baseline_result)
    
    results = []
    
    for component_name in components.keys():
        # Create ablated configuration (remove this component)
        ablated_components = components.copy()
        ablated_components[component_name] = None  # or False, depending on usage
        
        # Run with ablated configuration
        try:
            ablated_result = baseline_func(ablated_components)
            ablated_performance = performance_metric(ablated_result)
        except Exception as e:
            warnings.warn(f"Ablation failed for {component_name}: {e}")
            ablated_performance = np.nan
        
        # Compute delta
        delta = ablated_performance - baseline_performance
        delta_pct = (delta / baseline_performance * 100) if baseline_performance != 0 else np.nan
        
        results.append({
            "component": component_name,
            "performance": ablated_performance,
            "baseline_performance": baseline_performance,
            "delta": delta,
            "delta_pct": delta_pct,
            "importance": abs(delta),
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    
    return df


def component_importance_ranking(
    ablation_results: pd.DataFrame,
) -> pd.DataFrame:
    """Rank components by importance from ablation study results.
    
    Args:
        ablation_results: DataFrame from ablation_study()
        
    Returns:
        DataFrame with ranked components:
            - rank: importance rank (1 = most important)
            - component: component name
            - importance: absolute impact
            - impact_direction: "positive" if removing hurts, "negative" if removing helps
            
    Example:
        >>> ranking = component_importance_ranking(ablation_results)
        >>> print(ranking.head(3))  # Top 3 most important components
    """
    df = ablation_results.copy()
    
    # Determine impact direction
    df["impact_direction"] = df["delta"].apply(
        lambda x: "negative" if x < 0 else "positive" if x > 0 else "neutral"
    )
    
    # Rank by importance
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    
    # Select relevant columns
    df = df[["rank", "component", "importance", "delta", "delta_pct", "impact_direction"]]
    
    return df


def interaction_effects(
    baseline_func: Callable,
    components: Dict[str, Any],
    performance_metric: Callable[[Any], float],
    max_interaction_order: int = 2,
) -> pd.DataFrame:
    """Analyze interaction effects between components.
    
    Tests whether removing multiple components simultaneously has
    synergistic or antagonistic effects.
    
    Args:
        baseline_func: Function that runs experiment
        components: Dict mapping component names to values
        performance_metric: Function extracting performance
        max_interaction_order: Maximum number of components to remove together
            (2 = pairwise, 3 = three-way, etc.)
            
    Returns:
        DataFrame with interaction analysis:
            - components_removed: tuple of component names
            - performance: performance with these components removed
            - expected_additive: expected performance if effects were additive
            - interaction: performance - expected_additive
            - interaction_type: "synergistic" or "antagonistic"
            
    Example:
        >>> interactions = interaction_effects(run_exp, components, get_sharpe, max_interaction_order=2)
        >>> print(interactions[interactions['interaction'].abs() > 0.1])
    """
    # Get baseline
    baseline_result = baseline_func(components)
    baseline_performance = performance_metric(baseline_result)
    
    # Get individual effects
    individual_effects = {}
    for component_name in components.keys():
        ablated_components = components.copy()
        ablated_components[component_name] = None
        try:
            result = baseline_func(ablated_components)
            perf = performance_metric(result)
            individual_effects[component_name] = perf - baseline_performance
        except Exception as e:
            warnings.warn(f"Failed for {component_name}: {e}")
            individual_effects[component_name] = 0.0
    
    # Test combinations
    component_names = list(components.keys())
    results = []
    
    for order in range(2, min(max_interaction_order + 1, len(component_names) + 1)):
        for combo in combinations(component_names, order):
            # Remove all components in combo
            ablated_components = components.copy()
            for comp in combo:
                ablated_components[comp] = None
            
            try:
                result = baseline_func(ablated_components)
                actual_perf = performance_metric(result)
            except Exception as e:
                warnings.warn(f"Failed for {combo}: {e}")
                continue
            
            # Expected performance if effects were additive
            expected_delta = sum(individual_effects[comp] for comp in combo)
            expected_perf = baseline_performance + expected_delta
            
            # Interaction effect
            actual_delta = actual_perf - baseline_performance
            interaction = actual_delta - expected_delta
            
            interaction_type = "neutral"
            if abs(interaction) > 0.01:  # threshold
                if interaction < 0:
                    interaction_type = "synergistic_negative"  # worse than expected
                else:
                    interaction_type = "synergistic_positive"  # better than expected
            
            results.append({
                "components_removed": combo,
                "n_components": len(combo),
                "performance": actual_perf,
                "expected_additive": expected_perf,
                "interaction": interaction,
                "interaction_type": interaction_type,
            })
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("interaction", key=abs, ascending=False).reset_index(drop=True)
    
    return df


def marginal_contribution(
    baseline_func: Callable,
    components: Dict[str, Any],
    performance_metric: Callable[[Any], float],
) -> pd.DataFrame:
    """Compute marginal contribution of each component.
    
    Marginal contribution = performance gain from adding this component
    to an empty system.
    
    Args:
        baseline_func: Function that runs experiment
        components: Dict mapping component names to values
        performance_metric: Function extracting performance
        
    Returns:
        DataFrame with marginal contributions:
            - component: component name
            - with_component: performance with this component only
            - without_any: performance with no components
            - marginal_contribution: with_component - without_any
            - contribution_pct: percentage of total performance
            
    Example:
        >>> marginal = marginal_contribution(run_exp, components, get_sharpe)
        >>> print(marginal.sort_values('marginal_contribution', ascending=False))
    """
    # Performance with no components
    no_components = {k: None for k in components.keys()}
    try:
        no_comp_result = baseline_func(no_components)
        no_comp_performance = performance_metric(no_comp_result)
    except Exception as e:
        warnings.warn(f"No-component baseline failed: {e}")
        no_comp_performance = 0.0
    
    # Full system performance
    full_result = baseline_func(components)
    full_performance = performance_metric(full_result)
    
    results = []
    
    for component_name, component_value in components.items():
        # System with only this component
        single_comp = no_components.copy()
        single_comp[component_name] = component_value
        
        try:
            result = baseline_func(single_comp)
            single_comp_performance = performance_metric(result)
        except Exception as e:
            warnings.warn(f"Failed for {component_name} alone: {e}")
            single_comp_performance = no_comp_performance
        
        marginal = single_comp_performance - no_comp_performance
        
        # Contribution as percentage of full system
        if full_performance != 0:
            contribution_pct = (marginal / full_performance) * 100
        else:
            contribution_pct = 0.0
        
        results.append({
            "component": component_name,
            "with_component": single_comp_performance,
            "without_any": no_comp_performance,
            "marginal_contribution": marginal,
            "contribution_pct": contribution_pct,
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("marginal_contribution", ascending=False).reset_index(drop=True)
    
    return df


def full_factorial_design(
    baseline_func: Callable,
    components: Dict[str, List[Any]],
    performance_metric: Callable[[Any], float],
    n_max_combinations: Optional[int] = None,
) -> pd.DataFrame:
    """Run full factorial design over component configurations.
    
    Warning: Exponential in number of components. Use n_max_combinations
    to limit computational cost.
    
    Args:
        baseline_func: Function that runs experiment
        components: Dict mapping component names to lists of possible values
            Example: {"k_clusters": [3, 5, 8], "embedding": ["tfidf", "transformer"]}
        performance_metric: Function extracting performance
        n_max_combinations: Maximum number of combinations to test (None = all)
        
    Returns:
        DataFrame with all tested configurations and their performance
        
    Example:
        >>> components = {
        ...     "k_clusters": [3, 5, 8],
        ...     "alpha": [0.01, 0.05, 0.1],
        ... }
        >>> factorial = full_factorial_design(run_exp, components, get_sharpe)
        >>> print(factorial.sort_values('performance', ascending=False).head())
    """
    from itertools import product
    
    # Generate all combinations
    component_names = list(components.keys())
    component_values = [components[name] for name in component_names]
    all_combinations = list(product(*component_values))
    
    # Limit if requested
    if n_max_combinations is not None and len(all_combinations) > n_max_combinations:
        warnings.warn(
            f"Full factorial has {len(all_combinations)} combinations. "
            f"Sampling {n_max_combinations} randomly."
        )
        indices = np.random.choice(len(all_combinations), n_max_combinations, replace=False)
        all_combinations = [all_combinations[i] for i in indices]
    
    results = []
    
    for combo in all_combinations:
        # Build configuration
        config = dict(zip(component_names, combo))
        
        # Run experiment
        try:
            result = baseline_func(config)
            performance = performance_metric(result)
        except Exception as e:
            warnings.warn(f"Failed for {config}: {e}")
            performance = np.nan
        
        # Record result
        row = config.copy()
        row["performance"] = performance
        results.append(row)
    
    df = pd.DataFrame(results)
    df = df.sort_values("performance", ascending=False).reset_index(drop=True)
    
    return df


def visualize_ablation_results(
    ablation_results: pd.DataFrame,
    title: str = "Component Importance (Ablation Study)",
) -> None:
    """Visualize ablation study results as a bar chart.
    
    Args:
        ablation_results: DataFrame from ablation_study()
        title: Plot title
        
    Example:
        >>> visualize_ablation_results(ablation_results)
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    
    df = ablation_results.sort_values("importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
    
    colors = ["red" if d < 0 else "green" for d in df["delta"]]
    
    ax.barh(df["component"], df["delta"], color=colors, alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Performance Change (Δ)")
    ax.set_ylabel("Component Removed")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    
    fig.tight_layout()


def generate_ablation_report(
    ablation_results: pd.DataFrame,
    interaction_results: Optional[pd.DataFrame] = None,
) -> str:
    """Generate a text report summarizing ablation study findings.
    
    Args:
        ablation_results: DataFrame from ablation_study()
        interaction_results: Optional DataFrame from interaction_effects()
        
    Returns:
        Formatted string report
        
    Example:
        >>> report = generate_ablation_report(ablation_results, interaction_results)
        >>> print(report)
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ABLATION STUDY REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Component importance
    ranking = component_importance_ranking(ablation_results)
    lines.append("### Component Importance Ranking ###")
    lines.append("")
    for _, row in ranking.iterrows():
        symbol = "↓" if row["impact_direction"] == "negative" else "↑"
        lines.append(
            f"{row['rank']}. {row['component']}: "
            f"Δ={row['delta']:.4f} ({row['delta_pct']:.1f}%) {symbol}"
        )
    lines.append("")
    
    # Most critical component
    most_important = ranking.iloc[0]
    lines.append(f"Most critical component: {most_important['component']}")
    lines.append(f"  Removing it changes performance by {most_important['delta']:.4f}")
    lines.append("")
    
    # Interaction effects
    if interaction_results is not None and len(interaction_results) > 0:
        lines.append("### Notable Interaction Effects ###")
        lines.append("")
        significant_interactions = interaction_results[
            interaction_results["interaction"].abs() > 0.05
        ]
        if len(significant_interactions) > 0:
            for _, row in significant_interactions.head(5).iterrows():
                components_str = ", ".join(row["components_removed"])
                lines.append(
                    f"{components_str}: interaction={row['interaction']:.4f} "
                    f"({row['interaction_type']})"
                )
        else:
            lines.append("No significant interactions detected.")
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)

