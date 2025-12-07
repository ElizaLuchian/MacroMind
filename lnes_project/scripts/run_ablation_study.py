"""Script to run ablation study on system components.

This script systematically removes components to understand their importance.

Usage:
    python scripts/run_ablation_study.py --dataset small
    python scripts/run_ablation_study.py --dataset fnspid --tickers AAPL
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
from src import ablation


def run_experiment_with_components(components: Dict[str, Any]) -> Dict[str, Any]:
    """Run experiment with specified components.
    
    Args:
        components: Dict specifying which components to include
        
    Returns:
        Experiment result dictionary
    """
    # Extract configuration from components
    dataset = components.get("dataset", "small")
    backend = components.get("embedding_backend", "tfidf")
    use_clustering = components.get("use_clustering", True)
    tickers = components.get("tickers", ["AAPL"])
    
    # Configure embedder
    embedder_kwargs = {"backend": backend if backend is not None else "tfidf"}
    
    # Run experiment
    if dataset == "small":
        result = run_small_dataset_experiment(embedder_kwargs=embedder_kwargs)
    else:
        result = run_fnspid_experiment(
            tickers=tickers if tickers is not None else ["AAPL"],
            embedder_kwargs=embedder_kwargs,
        )
    
    return result


def extract_performance(result: Dict[str, Any]) -> float:
    """Extract primary performance metric from result.
    
    Args:
        result: Experiment result
        
    Returns:
        Performance score (Sharpe-like metric)
    """
    metrics = result.get("metrics", {})
    
    # Try to get directional accuracy
    if "directional_accuracy" in metrics:
        return metrics["directional_accuracy"]
    
    # Fallback: compute from simulation
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
        description="Run ablation study on system components"
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
        "--output-report",
        type=Path,
        default=None,
        help="Save ablation report to file",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    print()
    
    # Define components to ablate
    components = {
        "dataset": args.dataset,
        "embedding_backend": "tfidf",
        "use_clustering": True,
        "tickers": args.fnspid_tickers,
    }
    
    print("Components to test:")
    for comp_name, comp_value in components.items():
        print(f"  - {comp_name}: {comp_value}")
    print()
    
    # Run ablation study
    print("Running ablation study (this may take a while)...")
    print()
    
    ablation_results = ablation.ablation_study(
        baseline_func=run_experiment_with_components,
        components=components,
        performance_metric=extract_performance,
    )
    
    print("✓ Ablation study completed")
    print()
    
    # Display results
    print("=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print()
    
    print(ablation_results.to_string(index=False))
    print()
    
    # Component importance ranking
    print("=" * 70)
    print("COMPONENT IMPORTANCE RANKING")
    print("=" * 70)
    print()
    
    ranking = ablation.component_importance_ranking(ablation_results)
    print(ranking.to_string(index=False))
    print()
    
    # Generate report
    print("=" * 70)
    print("ABLATION REPORT")
    print("=" * 70)
    print()
    
    report = ablation.generate_ablation_report(ablation_results)
    print(report)
    
    # Save report
    if args.output_report:
        args.output_report.write_text(report)
        print(f"✓ Report saved to: {args.output_report}")
        print()
    
    print("=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

