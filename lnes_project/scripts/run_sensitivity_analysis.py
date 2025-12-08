"""Script to run sensitivity analysis on hyperparameters.

This script varies hyperparameters and measures impact on performance.

Usage:
    python scripts/run_sensitivity_analysis.py --dataset small
    python scripts/run_sensitivity_analysis.py --dataset fnspid --tickers AAPL --method morris
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_smallset import run_small_dataset_experiment
from src.experiment_fnspid import run_fnspid_experiment
from src import sensitivity


def run_experiment_with_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run experiment with specified parameters.
    
    Args:
        params: Parameter configuration
        
    Returns:
        Experiment result
    """
    dataset = params.get("dataset", "small")
    backend = params.get("backend", "tfidf")
    tickers = params.get("tickers", ["AAPL"])
    
    embedder_kwargs = {"backend": backend}
    
    if dataset == "small":
        result = run_small_dataset_experiment(embedder_kwargs=embedder_kwargs)
    else:
        result = run_fnspid_experiment(
            tickers=tickers,
            embedder_kwargs=embedder_kwargs,
        )
    
    return result


def extract_performance(result: Dict[str, Any]) -> float:
    """Extract performance metric from result.
    
    Args:
        result: Experiment result
        
    Returns:
        Performance score
    """
    metrics = result.get("metrics", {})
    
    if "directional_accuracy" in metrics:
        return metrics["directional_accuracy"]
    
    # Fallback
    simulation = result.get("simulation")
    if simulation is not None:
        sim_frame = simulation.to_frame()
        if "simulated_close" in sim_frame.columns:
            prices = sim_frame["simulated_close"].values
            returns = np.diff(prices) / prices[:-1]
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
            return sharpe
    
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis on hyperparameters"
    )
    parser.add_argument(
        "--dataset",
        choices=["small", "fnspid"],
        default="small",
        help="Dataset to use",
    )
    parser.add_argument(
        "--fnspid-tickers",
        nargs="+",
        default=["AAPL"],
        help="Ticker symbols for FNSPID dataset",
    )
    parser.add_argument(
        "--method",
        choices=["oat", "morris"],
        default="oat",
        help="Sensitivity analysis method (oat=one-at-a-time, morris=Morris screening)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Save sensitivity report to file",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)
    print()
    
    # Define baseline parameters
    baseline_params = {
        "dataset": args.dataset,
        "backend": "tfidf",
        "tickers": args.fnspid_tickers,
    }
    
    print("Baseline parameters:")
    for param_name, param_value in baseline_params.items():
        print(f"  {param_name}: {param_value}")
    print()
    
    if args.method == "oat":
        print("Running one-at-a-time sensitivity analysis...")
        print()
        
        # Define parameter ranges to test
        parameter_ranges = {
            "backend": ["tfidf", "sentence-transformer"],
        }
        
        if args.dataset == "fnspid":
            # Could add more parameters specific to FNSPID
            pass
        
        # Run OAT sensitivity
        sensitivity_results = sensitivity.one_at_a_time_sensitivity(
            baseline_func=run_experiment_with_params,
            parameters=baseline_params,
            parameter_ranges=parameter_ranges,
            performance_metric=extract_performance,
        )
        
        print("✓ Sensitivity analysis completed")
        print()
        
        # Display results
        print("=" * 70)
        print("SENSITIVITY RESULTS")
        print("=" * 70)
        print()
        print(sensitivity_results.to_string(index=False))
        print()
        
        # Generate report
        report = sensitivity.generate_sensitivity_report(sensitivity_results)
        print(report)
    
    elif args.method == "morris":
        print("Running Morris screening method...")
        print()
        
        # Define parameter ranges for Morris (numeric parameters only)
        # For categorical parameters, we'd need a different approach
        # Here we use a simplified example
        
        print("Note: Morris screening requires numeric parameters.")
        print("Using synthetic numeric parameters for demonstration.")
        print()
        
        # Synthetic parameter ranges
        parameter_ranges = {
            "alpha": (0.001, 0.1),
            "noise_std": (0.0, 0.05),
        }
        
        # Modified experiment function that uses these parameters
        def run_exp_numeric(params):
            # In a real scenario, these would be passed to the simulation
            # For now, we just run with defaults
            return run_experiment_with_params(baseline_params)
        
        morris_results = sensitivity.morris_screening(
            baseline_func=run_exp_numeric,
            parameter_ranges=parameter_ranges,
            performance_metric=extract_performance,
            n_trajectories=10,
        )
        
        print("✓ Morris screening completed")
        print()
        
        # Display results
        print("=" * 70)
        print("MORRIS SENSITIVITY INDICES")
        print("=" * 70)
        print()
        print(morris_results.to_string(index=False))
        print()
        
        # Generate report
        report = sensitivity.generate_sensitivity_report(
            pd.DataFrame(),  # Empty OAT results
            morris_results=morris_results,
        )
        print(report)
    
    # Save report
    if args.output_report:
        args.output_report.write_text(report if 'report' in locals() else "No report generated")
        print(f"✓ Report saved to: {args.output_report}")
        print()
    
    print("=" * 70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

