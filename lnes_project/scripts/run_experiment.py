"""Convenience script to run the small dataset experiment with visual output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clusterer import plot_cluster_distribution
from src.experiment_fnspid import run_fnspid_experiment
from src.experiment_smallset import run_small_dataset_experiment
from src import metrics as metrics_module


def _plot_prices(reference: pd.Series, simulated: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    reference = reference.sort_index()
    simulated = simulated.sort_index()
    ax.plot(reference.index, reference.values, label="Reference close", color="#4C72B0")
    ax.plot(simulated.index, simulated.values, label="Simulated close", color="#DD8452", linestyle="--")
    ax.set_title("Reference vs Simulated Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()


def _plot_agent_profitability(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["agent"], df["pnl"], color="#55A868")
    ax.set_title("Agent Profitability (P&L)")
    ax.set_ylabel("P&L (arbitrary units)")
    fig.tight_layout()


def _create_action_log_dataframe(simulation, merged: pd.DataFrame) -> pd.DataFrame:
    """Create a detailed DataFrame showing agent actions alongside market context."""
    sim_frame = simulation.to_frame()
    merged_with_date = merged.copy()
    merged_with_date["date"] = pd.to_datetime(merged_with_date["date"])
    
    # Select available columns from merged data
    available_cols = ["date", "close", "cluster_id"]
    if "headline" in merged_with_date.columns:
        available_cols.append("headline")
    elif "combined_text" in merged_with_date.columns:
        available_cols.append("combined_text")
    
    # Merge simulation results with original data
    action_df = pd.merge(
        sim_frame,
        merged_with_date[available_cols].copy(),
        on="date",
        how="left",
    )
    
    # Reorder columns for readability (only include columns that exist)
    base_cols = ["date", "close", "simulated_close", "order_flow"]
    if "cluster_id" in action_df.columns:
        base_cols.append("cluster_id")
    if "headline" in action_df.columns:
        base_cols.append("headline")
    elif "combined_text" in action_df.columns:
        base_cols.append("combined_text")
    
    action_cols = [col for col in action_df.columns if col.startswith("action_")]
    final_cols = [col for col in base_cols + action_cols if col in action_df.columns]
    action_df = action_df[final_cols]
    
    return action_df


def _plot_agent_actions(action_df: pd.DataFrame) -> None:
    """Visualize agent action patterns over time."""
    action_cols = [col for col in action_df.columns if col.startswith("action_")]
    if not action_cols:
        return
    
    fig, axes = plt.subplots(len(action_cols), 1, figsize=(12, 2.5 * len(action_cols)), sharex=True)
    if len(action_cols) == 1:
        axes = [axes]
    
    action_map = {"buy": 1, "sell": -1, "hold": 0}
    
    for idx, col in enumerate(action_cols):
        ax = axes[idx]
        agent_name = col.replace("action_", "")
        actions = action_df[col].map(action_map)
        ax.plot(action_df["date"], actions, marker="o", markersize=3, linestyle="-", linewidth=0.5)
        ax.set_ylabel(agent_name, rotation=0, ha="right")
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["sell", "hold", "buy"])
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
    
    axes[-1].set_xlabel("Date")
    fig.suptitle("Agent Actions Over Time", y=0.995)
    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Latent News Event Simulation experiment with visuals.")
    parser.add_argument(
        "--dataset",
        choices=["small", "fnspid"],
        default="small",
        help="Dataset to run the experiment on.",
    )
    parser.add_argument(
        "--backend",
        choices=["sentence-transformer", "tfidf"],
        default="sentence-transformer",
        help="Embedding backend to use.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional path to a directory containing small_news.csv and small_prices.csv.",
    )
    parser.add_argument(
        "--fnspid-tickers",
        nargs="+",
        default=None,
        help="Ticker symbols (required when --dataset fnspid).",
    )
    parser.add_argument("--fnspid-start-date", default=None, help="Inclusive start date for FNSPID slice (YYYY-MM-DD).")
    parser.add_argument("--fnspid-end-date", default=None, help="Inclusive end date for FNSPID slice (YYYY-MM-DD).")
    parser.add_argument(
        "--fnspid-news-limit",
        type=int,
        default=5000,
        help="Number of news rows to request from FNSPID (via split slicing).",
    )
    parser.add_argument(
        "--fnspid-price-limit",
        type=int,
        default=20000,
        help="Number of price rows to request from FNSPID (via split slicing).",
    )
    parser.add_argument(
        "--fnspid-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory for FNSPID downloads.",
    )
    parser.add_argument(
        "--fnspid-auth-token",
        default=None,
        help="Optional Hugging Face auth token if the dataset requires authentication.",
    )
    parser.add_argument(
        "--fnspid-local-news",
        type=Path,
        default=None,
        help="Optional local CSV path for the FNSPID news table (skips Hugging Face download).",
    )
    parser.add_argument(
        "--fnspid-local-prices",
        type=Path,
        default=None,
        help="Optional local CSV path for the FNSPID price table (skips Hugging Face download).",
    )
    parser.add_argument(
        "--show-actions",
        action="store_true",
        help="Display detailed agent action log in the console.",
    )
    parser.add_argument(
        "--save-actions-csv",
        type=Path,
        default=None,
        help="Save agent action log to a CSV file (e.g., actions.csv).",
    )
    parser.add_argument(
        "--plot-actions",
        action="store_true",
        help="Generate a plot showing agent actions over time.",
    )
    parser.add_argument(
        "--advanced-metrics",
        action="store_true",
        help="Compute and display advanced financial metrics (Sharpe, drawdown, VaR, etc.).",
    )
    parser.add_argument(
        "--transaction-costs",
        action="store_true",
        help="Apply transaction costs to agent profitability analysis.",
    )
    parser.add_argument(
        "--fixed-cost",
        type=float,
        default=0.01,
        help="Fixed transaction cost per trade (default: 0.01).",
    )
    parser.add_argument(
        "--proportional-cost",
        type=float,
        default=0.001,
        help="Proportional transaction cost (default: 0.001 = 10 bps).",
    )
    args = parser.parse_args()

    embedder_kwargs = {"backend": args.backend}
    if args.dataset == "small":
        result = run_small_dataset_experiment(data_dir=args.data_dir, embedder_kwargs=embedder_kwargs)
    else:
        if not args.fnspid_tickers:
            parser.error("--fnspid-tickers is required when --dataset fnspid")
        result = run_fnspid_experiment(
            tickers=args.fnspid_tickers,
            start_date=args.fnspid_start_date,
            end_date=args.fnspid_end_date,
            news_limit=args.fnspid_news_limit,
            price_limit=args.fnspid_price_limit,
            embedder_kwargs=embedder_kwargs,
            cache_dir=args.fnspid_cache_dir,
            auth_token=args.fnspid_auth_token,
            local_news_csv=args.fnspid_local_news,
            local_price_csv=args.fnspid_local_prices,
        )

    simulation = result["simulation"]
    metrics = result["metrics"]
    clusters = result["clusters"]
    merged = result["merged"]

    print("=== Simulation Summary ===")
    print(f"Directional accuracy: {metrics['directional_accuracy']:.3f}")
    print(f"Volatility clustering (lag 5): {metrics['volatility_clustering']:.3f}")
    print(f"Cluster-price correlation: {metrics['cluster_price_correlation']:.3f}")
    print("\nAgent profitability:")
    print(metrics["agent_profitability"].to_string(index=False))
    
    # Advanced metrics if requested
    if args.advanced_metrics:
        print("\n" + "=" * 70)
        print("=== ADVANCED FINANCIAL METRICS ===")
        print("=" * 70)
        
        # Get price series
        sim_prices = simulation.to_frame()["simulated_close"].values
        ref_prices = merged["close"].values
        
        # Generate comprehensive report
        report = metrics_module.generate_performance_report(
            prices=sim_prices,
            action_log=simulation.action_log,
            reference_prices=ref_prices,
            risk_free_rate=0.02,
            include_transaction_costs=args.transaction_costs,
            fixed_cost=args.fixed_cost if args.transaction_costs else 0.0,
            proportional_cost=args.proportional_cost if args.transaction_costs else 0.0,
        )
        
        # Print formatted report
        formatted = metrics_module.format_performance_report(report, title="Simulated Portfolio Performance")
        print(formatted)
        
        # Print agent comparison if available
        if "agent_profitability" in report:
            print("\n" + "=" * 70)
            print("=== DETAILED AGENT COMPARISON ===")
            print("=" * 70)
            comp_df = metrics_module.compare_agents_report(
                simulation.action_log,
                sim_prices,
            )
            print(comp_df.to_string(index=False))
        
        # Print transaction cost analysis if enabled
        if args.transaction_costs and "transaction_costs" in report:
            print("\n" + "=" * 70)
            print("=== TRANSACTION COST ANALYSIS ===")
            print("=" * 70)
            print(report["transaction_costs"].to_string(index=False))

    # Create detailed action log DataFrame
    action_df = _create_action_log_dataframe(simulation, merged)

    if args.show_actions:
        print("\n=== Agent Action Log (first 20 rows) ===")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 50)
        print(action_df.head(20).to_string(index=False))
        if len(action_df) > 20:
            print(f"\n... ({len(action_df) - 20} more rows)")

    if args.save_actions_csv:
        output_path = Path(args.save_actions_csv)
        action_df.to_csv(output_path, index=False)
        print(f"\n=== Agent actions saved to: {output_path} ===")

    sim_frame = simulation.to_frame().set_index("date")
    simulated_prices = sim_frame["simulated_close"]
    reference_prices = merged.set_index("date")["close"]
    _plot_prices(reference=reference_prices, simulated=simulated_prices)
    plot_cluster_distribution(clusters.labels)
    _plot_agent_profitability(metrics["agent_profitability"])

    if args.plot_actions:
        _plot_agent_actions(action_df)

    plt.show()


if __name__ == "__main__":
    main()

