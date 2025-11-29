"""Sensitivity analysis tools for parameter robustness assessment.

This module provides methods for analyzing how sensitive model performance
is to changes in hyperparameters and configuration choices.

Functions:
    one_at_a_time: Simple one-parameter-at-a-time sensitivity
    sobol_indices: Variance-based global sensitivity (Sobol indices)
    morris_screening: Morris method for factor screening
    parameter_sweep: Sweep parameters over ranges
    sensitivity_heatmap: Visualize 2D sensitivity
"""

from __future__ import annotations

from typing import Dict, List, Callable, Any, Optional, Tuple
import warnings

import numpy as np
import pandas as pd


def one_at_a_time_sensitivity(
    baseline_func: Callable,
    parameters: Dict[str, Any],
    parameter_ranges: Dict[str, List[Any]],
    performance_metric: Callable[[Any], float],
) -> pd.DataFrame:
    """One-at-a-time (OAT) sensitivity analysis.
    
    Varies one parameter at a time while keeping others at baseline values.
    Simple but doesn't capture interaction effects.
    
    Args:
        baseline_func: Function that runs experiment
            Signature: baseline_func(params_dict) -> result
        parameters: Baseline parameter configuration
        parameter_ranges: Dict mapping parameter names to lists of values to test
        performance_metric: Function extracting performance from result
        
    Returns:
        DataFrame with sensitivity results:
            - parameter: parameter name
            - value: parameter value tested
            - performance: resulting performance
            - baseline_performance: performance at baseline
            - sensitivity: (performance - baseline) / (value - baseline_value)
            
    Example:
        >>> baseline = {"alpha": 0.01, "k_clusters": 3}
        >>> ranges = {"alpha": [0.001, 0.01, 0.05, 0.1], "k_clusters": [3, 5, 8, 10]}
        >>> sensitivity = one_at_a_time_sensitivity(run_exp, baseline, ranges, get_sharpe)
        >>> print(sensitivity)
    """
    # Compute baseline performance
    baseline_result = baseline_func(parameters)
    baseline_performance = performance_metric(baseline_result)
    
    results = []
    
    for param_name, param_values in parameter_ranges.items():
        baseline_value = parameters[param_name]
        
        for test_value in param_values:
            # Create test configuration
            test_params = parameters.copy()
            test_params[param_name] = test_value
            
            # Run experiment
            try:
                result = baseline_func(test_params)
                performance = performance_metric(result)
            except Exception as e:
                warnings.warn(f"Failed for {param_name}={test_value}: {e}")
                performance = np.nan
            
            # Compute sensitivity (derivative approximation)
            if isinstance(test_value, (int, float)) and isinstance(baseline_value, (int, float)):
                if test_value != baseline_value:
                    sensitivity = (performance - baseline_performance) / (test_value - baseline_value)
                else:
                    sensitivity = 0.0
            else:
                sensitivity = np.nan
            
            results.append({
                "parameter": param_name,
                "value": test_value,
                "performance": performance,
                "baseline_performance": baseline_performance,
                "baseline_value": baseline_value,
                "delta_performance": performance - baseline_performance,
                "sensitivity": sensitivity,
            })
    
    df = pd.DataFrame(results)
    return df


def morris_screening(
    baseline_func: Callable,
    parameter_ranges: Dict[str, Tuple[float, float]],
    performance_metric: Callable[[Any], float],
    n_trajectories: int = 10,
    n_levels: int = 4,
) -> pd.DataFrame:
    """Morris method for global sensitivity screening.
    
    Efficient method for identifying important parameters using elementary effects.
    Good for large parameter spaces.
    
    Args:
        baseline_func: Function that runs experiment
        parameter_ranges: Dict mapping parameter names to (min, max) tuples
        performance_metric: Function extracting performance
        n_trajectories: Number of random trajectories to sample
        n_levels: Number of levels for each parameter
        
    Returns:
        DataFrame with Morris indices:
            - parameter: parameter name
            - mu: mean absolute elementary effect (overall influence)
            - sigma: standard deviation of elementary effects (interactions/nonlinearity)
            - mu_star: mean of absolute elementary effects
            
    Example:
        >>> ranges = {"alpha": (0.001, 0.1), "k_clusters": (3, 10)}
        >>> morris = morris_screening(run_exp, ranges, get_sharpe, n_trajectories=20)
        >>> print(morris.sort_values('mu_star', ascending=False))
    """
    param_names = list(parameter_ranges.keys())
    n_params = len(param_names)
    
    # Generate Morris design
    trajectories = []
    elementary_effects = {name: [] for name in param_names}
    
    for _ in range(n_trajectories):
        # Random starting point
        base_point = {}
        for name, (min_val, max_val) in parameter_ranges.items():
            level = np.random.randint(0, n_levels)
            base_point[name] = min_val + (max_val - min_val) * level / (n_levels - 1)
        
        # Compute base performance
        try:
            base_result = baseline_func(base_point)
            base_perf = performance_metric(base_result)
        except Exception as e:
            warnings.warn(f"Base point failed: {e}")
            continue
        
        # Permute parameters
        perm = np.random.permutation(param_names)
        
        current_point = base_point.copy()
        current_perf = base_perf
        
        for param_name in perm:
            # Move one step in this parameter
            min_val, max_val = parameter_ranges[param_name]
            delta = (max_val - min_val) / (n_levels - 1)
            
            # Random direction
            direction = 1 if np.random.rand() > 0.5 else -1
            new_value = current_point[param_name] + direction * delta
            new_value = np.clip(new_value, min_val, max_val)
            
            new_point = current_point.copy()
            new_point[param_name] = new_value
            
            try:
                new_result = baseline_func(new_point)
                new_perf = performance_metric(new_result)
            except Exception as e:
                warnings.warn(f"New point failed: {e}")
                continue
            
            # Elementary effect
            param_delta = new_value - current_point[param_name]
            if abs(param_delta) > 1e-10:
                ee = (new_perf - current_perf) / param_delta
                elementary_effects[param_name].append(ee)
            
            current_point = new_point
            current_perf = new_perf
    
    # Compute Morris indices
    results = []
    for param_name in param_names:
        effects = np.array(elementary_effects[param_name])
        if len(effects) > 0:
            mu = np.mean(effects)
            sigma = np.std(effects, ddof=1) if len(effects) > 1 else 0.0
            mu_star = np.mean(np.abs(effects))
        else:
            mu = sigma = mu_star = np.nan
        
        results.append({
            "parameter": param_name,
            "mu": mu,
            "sigma": sigma,
            "mu_star": mu_star,
            "n_effects": len(effects),
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("mu_star", ascending=False).reset_index(drop=True)
    
    return df


def parameter_sweep_2d(
    baseline_func: Callable,
    param1_name: str,
    param1_values: List[Any],
    param2_name: str,
    param2_values: List[Any],
    fixed_params: Dict[str, Any],
    performance_metric: Callable[[Any], float],
) -> pd.DataFrame:
    """2D parameter sweep for visualizing interactions.
    
    Args:
        baseline_func: Function that runs experiment
        param1_name: First parameter name
        param1_values: Values to test for param1
        param2_name: Second parameter name
        param2_values: Values to test for param2
        fixed_params: Fixed values for other parameters
        performance_metric: Function extracting performance
        
    Returns:
        DataFrame with grid of results:
            - param1_name: value of param1
            - param2_name: value of param2
            - performance: resulting performance
            
    Example:
        >>> sweep = parameter_sweep_2d(
        ...     run_exp, "alpha", [0.001, 0.01, 0.1],
        ...     "k_clusters", [3, 5, 8],
        ...     {"seed": 42}, get_sharpe
        ... )
        >>> pivot = sweep.pivot(param1_name, param2_name, "performance")
        >>> sns.heatmap(pivot)
    """
    results = []
    
    for val1 in param1_values:
        for val2 in param2_values:
            params = fixed_params.copy()
            params[param1_name] = val1
            params[param2_name] = val2
            
            try:
                result = baseline_func(params)
                performance = performance_metric(result)
            except Exception as e:
                warnings.warn(f"Failed for {param1_name}={val1}, {param2_name}={val2}: {e}")
                performance = np.nan
            
            results.append({
                param1_name: val1,
                param2_name: val2,
                "performance": performance,
            })
    
    return pd.DataFrame(results)


def stability_analysis(
    baseline_func: Callable,
    parameters: Dict[str, Any],
    parameter_ranges: Dict[str, Tuple[float, float]],
    performance_metric: Callable[[Any], float],
    n_samples: int = 100,
    noise_std: float = 0.1,
) -> pd.DataFrame:
    """Analyze performance stability under parameter perturbations.
    
    Adds random noise to parameters and observes performance variation.
    
    Args:
        baseline_func: Function that runs experiment
        parameters: Baseline parameter configuration
        parameter_ranges: Valid ranges for parameters (for clipping)
        performance_metric: Function extracting performance
        n_samples: Number of noisy samples to generate
        noise_std: Standard deviation of noise (as fraction of range)
        
    Returns:
        DataFrame with stability metrics:
            - parameter: parameter name
            - mean_performance: mean performance under noise
            - std_performance: standard deviation of performance
            - cv: coefficient of variation (std / mean)
            - stability_score: 1 / (1 + cv)
            
    Example:
        >>> stability = stability_analysis(
        ...     run_exp, baseline_params, param_ranges,
        ...     get_sharpe, n_samples=50
        ... )
        >>> print(stability.sort_values('stability_score', ascending=False))
    """
    # Baseline performance
    baseline_result = baseline_func(parameters)
    baseline_performance = performance_metric(baseline_result)
    
    # Generate noisy samples for each parameter
    param_stability = []
    
    for param_name in parameters.keys():
        if param_name not in parameter_ranges:
            continue
        
        min_val, max_val = parameter_ranges[param_name]
        param_range = max_val - min_val
        
        perturbed_performances = []
        
        for _ in range(n_samples):
            # Add noise to this parameter
            noise = np.random.normal(0, noise_std * param_range)
            perturbed_value = parameters[param_name] + noise
            perturbed_value = np.clip(perturbed_value, min_val, max_val)
            
            # Run with perturbed parameter
            perturbed_params = parameters.copy()
            perturbed_params[param_name] = perturbed_value
            
            try:
                result = baseline_func(perturbed_params)
                perf = performance_metric(result)
                perturbed_performances.append(perf)
            except Exception as e:
                warnings.warn(f"Perturbation failed for {param_name}: {e}")
                continue
        
        if len(perturbed_performances) > 0:
            mean_perf = np.mean(perturbed_performances)
            std_perf = np.std(perturbed_performances, ddof=1)
            cv = std_perf / abs(mean_perf) if abs(mean_perf) > 1e-10 else np.inf
            stability_score = 1.0 / (1.0 + cv)
        else:
            mean_perf = std_perf = cv = np.nan
            stability_score = 0.0
        
        param_stability.append({
            "parameter": param_name,
            "baseline_performance": baseline_performance,
            "mean_performance": mean_perf,
            "std_performance": std_perf,
            "cv": cv,
            "stability_score": stability_score,
            "n_samples": len(perturbed_performances),
        })
    
    df = pd.DataFrame(param_stability)
    df = df.sort_values("stability_score", ascending=False).reset_index(drop=True)
    
    return df


def visualize_sensitivity_1d(
    sensitivity_results: pd.DataFrame,
    parameter_name: str,
    title: Optional[str] = None,
) -> None:
    """Visualize 1D sensitivity results for a single parameter.
    
    Args:
        sensitivity_results: DataFrame from one_at_a_time_sensitivity()
        parameter_name: Parameter to plot
        title: Optional plot title
        
    Example:
        >>> visualize_sensitivity_1d(sensitivity_results, "alpha")
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    
    df = sensitivity_results[sensitivity_results["parameter"] == parameter_name]
    
    if len(df) == 0:
        warnings.warn(f"No data for parameter: {parameter_name}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df["value"], df["performance"], marker="o", linewidth=2, markersize=6)
    
    # Baseline line
    baseline_val = df["baseline_value"].iloc[0]
    baseline_perf = df["baseline_performance"].iloc[0]
    ax.axvline(baseline_val, color="red", linestyle="--", label="Baseline", alpha=0.7)
    ax.axhline(baseline_perf, color="red", linestyle="--", alpha=0.7)
    
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Performance")
    ax.set_title(title or f"Sensitivity to {parameter_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()


def visualize_sensitivity_heatmap(
    sweep_results: pd.DataFrame,
    param1_name: str,
    param2_name: str,
    title: Optional[str] = None,
) -> None:
    """Visualize 2D parameter sweep as heatmap.
    
    Args:
        sweep_results: DataFrame from parameter_sweep_2d()
        param1_name: First parameter name (y-axis)
        param2_name: Second parameter name (x-axis)
        title: Optional plot title
        
    Example:
        >>> visualize_sensitivity_heatmap(sweep_results, "alpha", "k_clusters")
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    
    # Pivot to matrix form
    pivot = sweep_results.pivot(index=param1_name, columns=param2_name, values="performance")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
    
    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{v:.3g}" for v in pivot.columns])
    ax.set_yticklabels([f"{v:.3g}" for v in pivot.index])
    
    ax.set_xlabel(param2_name)
    ax.set_ylabel(param1_name)
    ax.set_title(title or f"Sensitivity Heatmap: {param1_name} vs {param2_name}")
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Performance")
    
    fig.tight_layout()


def generate_sensitivity_report(
    sensitivity_results: pd.DataFrame,
    morris_results: Optional[pd.DataFrame] = None,
    stability_results: Optional[pd.DataFrame] = None,
) -> str:
    """Generate text report summarizing sensitivity analysis.
    
    Args:
        sensitivity_results: DataFrame from one_at_a_time_sensitivity()
        morris_results: Optional DataFrame from morris_screening()
        stability_results: Optional DataFrame from stability_analysis()
        
    Returns:
        Formatted string report
        
    Example:
        >>> report = generate_sensitivity_report(sens_results, morris_results)
        >>> print(report)
    """
    lines = []
    lines.append("=" * 70)
    lines.append("SENSITIVITY ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # OAT sensitivity
    lines.append("### One-at-a-Time Sensitivity ###")
    lines.append("")
    for param in sensitivity_results["parameter"].unique():
        param_data = sensitivity_results[sensitivity_results["parameter"] == param]
        max_sens = param_data["sensitivity"].abs().max()
        lines.append(f"{param}: max sensitivity = {max_sens:.4f}")
    lines.append("")
    
    # Morris screening
    if morris_results is not None:
        lines.append("### Morris Screening (Important Parameters) ###")
        lines.append("")
        for _, row in morris_results.head(5).iterrows():
            lines.append(
                f"{row['parameter']}: μ*={row['mu_star']:.4f}, σ={row['sigma']:.4f}"
            )
        lines.append("")
    
    # Stability
    if stability_results is not None:
        lines.append("### Parameter Stability ###")
        lines.append("")
        for _, row in stability_results.iterrows():
            lines.append(
                f"{row['parameter']}: stability score = {row['stability_score']:.4f} "
                f"(CV={row['cv']:.4f})"
            )
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)

