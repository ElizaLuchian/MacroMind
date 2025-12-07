"""Script to run comprehensive agent comparison study.

This script runs multiple agents on the same dataset, performs statistical
comparisons, and generates a detailed comparison report.

Usage:
    python scripts/run_comparison_study.py --dataset small
    python scripts/run_comparison_study.py --dataset fnspid --tickers AAPL --save-db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_smallset import run_small_dataset_experiment
from src.experiment_fnspid import run_fnspid_experiment
from src import agent_comparison, statistical_tests, benchmarks
from src.results_db import ResultsDB


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive agent comparison study"
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
        "--fnspid-start-date",
        default=None,
        help="Start date for FNSPID (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--fnspid-end-date",
        default=None,
        help="End date for FNSPID (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--backend",
        choices=["sentence-transformer", "tfidf"],
        default="tfidf",
        help="Embedding backend",
    )
    parser.add_argument(
        "--metric",
        default="sharpe_ratio",
        help="Primary metric for comparison",
    )
    parser.add_argument(
        "--test-type",
        choices=["ttest", "wilcoxon"],
        default="ttest",
        help="Statistical test type",
    )
    parser.add_argument(
        "--correction",
        choices=["bonferroni", "holm", "fdr_bh"],
        default="holm",
        help="Multiple comparison correction method",
    )
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Save results to database",
    )
    parser.add_argument(
        "--db-path",
        default="experiments.db",
        help="Database path",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Save text report to file",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AGENT COMPARISON STUDY")
    print("=" * 70)
    print()
    
    # Run experiment
    print(f"Running experiment on {args.dataset} dataset...")
    embedder_kwargs = {"backend": args.backend}
    
    if args.dataset == "small":
        result = run_small_dataset_experiment(embedder_kwargs=embedder_kwargs)
    else:
        result = run_fnspid_experiment(
            tickers=args.fnspid_tickers,
            start_date=args.fnspid_start_date,
            end_date=args.fnspid_end_date,
            embedder_kwargs=embedder_kwargs,
        )
    
    simulation = result["simulation"]
    metrics_result = result["metrics"]
    
    print("✓ Experiment completed")
    print()
    
    # Extract agent performances
    print("Extracting agent performance data...")
    
    # Get agent names from action log
    action_log = simulation.action_log
    agent_names = [col.replace("action_", "") for col in action_log.columns 
                   if col.startswith("action_")]
    
    # Compute returns for each agent (simplified: sign of action * price change)
    sim_frame = simulation.to_frame()
    price_changes = sim_frame["simulated_close"].diff().fillna(0).values
    
    agent_performances = {}
    for agent_name in agent_names:
        action_col = f"action_{agent_name}"
        if action_col in action_log.columns:
            actions = action_log[action_col].map({"buy": 1, "sell": -1, "hold": 0}).fillna(0).values
            # Align with price_changes (action at t predicts change from t to t+1)
            if len(actions) == len(price_changes):
                agent_returns = actions * price_changes
            else:
                agent_returns = actions[:-1] * price_changes[1:]
            agent_performances[agent_name] = agent_returns
    
    print(f"✓ Extracted performance for {len(agent_performances)} agents")
    print()
    
    # Statistical comparison
    print("Performing pairwise statistical comparisons...")
    comparison_df = agent_comparison.compare_agents(
        agent_performances,
        test_type=args.test_type,
        correction_method=args.correction,
    )
    print(f"✓ Completed {len(comparison_df)} pairwise comparisons")
    print()
    
    # Rankings
    print("Computing agent rankings...")
    rankings = agent_comparison.rank_agents(agent_performances, metric_name=args.metric)
    print("✓ Rankings computed")
    print()
    
    # Benchmark evaluation
    print("Comparing against benchmarks...")
    benchmark_suite = benchmarks.BenchmarkSuite()
    benchmark_names = benchmark_suite.get_benchmark_names()
    
    # Get performances for agents that are also benchmarks
    benchmark_performances = {}
    for bench_name in benchmark_names:
        if bench_name in agent_performances:
            benchmark_performances[bench_name] = agent_performances[bench_name]
    
    if benchmark_performances:
        # Evaluate non-benchmark agents against benchmarks
        for agent_name, perf in agent_performances.items():
            if agent_name not in benchmark_names:
                eval_result = benchmarks.evaluate_against_benchmarks(
                    perf, agent_name, benchmark_performances, args.metric
                )
                print(f"  {agent_name} vs benchmarks:")
                for _, row in eval_result.head(3).iterrows():
                    print(f"    vs {row['benchmark']}: "
                          f"{'beats' if row['beats_benchmark'] else 'loses to'} "
                          f"(Δ={row['difference']:.4f})")
    print()
    
    # Generate comprehensive report
    print("Generating comparison report...")
    report = agent_comparison.summary_report(agent_performances, args.metric)
    print(report)
    
    # Save to database
    if args.save_db:
        print(f"Saving results to database: {args.db_path}")
        with ResultsDB(args.db_path) as db:
            # Save each agent's experiment
            experiment_ids = {}
            for agent_name, perf in agent_performances.items():
                config = {
                    "dataset": args.dataset,
                    "backend": args.backend,
                    "tickers": args.fnspid_tickers if args.dataset == "fnspid" else None,
                }
                
                performance = {
                    "mean_return": float(np.mean(perf)),
                    "std_return": float(np.std(perf)),
                    "sharpe_ratio": float(np.mean(perf) / np.std(perf)) if np.std(perf) > 0 else 0.0,
                    "win_rate": float(np.mean(perf > 0)),
                }
                
                exp_id = db.add_experiment(
                    agent_name=agent_name,
                    dataset=f"{args.dataset}_{'_'.join(args.fnspid_tickers)}" if args.dataset == "fnspid" else args.dataset,
                    config=config,
                    performance=performance,
                    experiment_type="comparison_study",
                )
                experiment_ids[agent_name] = exp_id
            
            # Save comparisons
            for _, row in comparison_df.iterrows():
                agent1 = row["agent1"]
                agent2 = row["agent2"]
                if agent1 in experiment_ids and agent2 in experiment_ids:
                    conclusion = "significant" if row["reject"] else "not significant"
                    db.add_comparison(
                        experiment_id_1=experiment_ids[agent1],
                        experiment_id_2=experiment_ids[agent2],
                        test_type=args.test_type,
                        test_statistic=row["test_statistic"],
                        p_value=row["p_value"],
                        effect_size=row["effect_size"],
                        conclusion=conclusion,
                    )
            
            print(f"✓ Saved {len(experiment_ids)} experiments and {len(comparison_df)} comparisons")
            print()
            
            # Database summary
            from src.results_db import create_results_summary
            summary = create_results_summary(db)
            print(summary)
    
    # Save report to file
    if args.output_report:
        args.output_report.write_text(report)
        print(f"✓ Report saved to: {args.output_report}")
    
    print()
    print("=" * 70)
    print("COMPARISON STUDY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

