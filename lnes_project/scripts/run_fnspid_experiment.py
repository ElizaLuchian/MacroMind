"""Run experiment with processed FNSPID data."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.agents import ContrarianAgent, MomentumAgent, NewsReactiveAgent, RandomAgent
from src.clusterer import cluster_embeddings, plot_cluster_distribution
from src.data_loader import load_news, load_prices, merge_news_and_prices
from src.embedder import NewsEmbedder
from src.metrics import summarize_metrics
from src.preprocess import preprocess_news
from src.simulator import simulate_market

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def plot_prices(reference: pd.Series, simulated: pd.Series, ticker: str) -> None:
    """Plot reference vs simulated prices."""
    fig, ax = plt.subplots(figsize=(14, 7))
    reference = reference.sort_index()
    simulated = simulated.sort_index()
    ax.plot(reference.index, reference.values, label=f"Real {ticker} Price", color="#4C72B0", linewidth=2)
    ax.plot(simulated.index, simulated.values, label="Simulated Price", color="#DD8452", linestyle="--", linewidth=2)
    ax.set_title(f"FNSPID Dataset: Real vs Simulated Prices for {ticker}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def plot_agent_profitability(df: pd.DataFrame) -> None:
    """Plot agent profitability."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#DD8452' if pnl < 0 else '#55A868' for pnl in df['pnl']]
    ax.bar(df["agent"], df["pnl"], color=colors)
    ax.set_title("Agent Profitability on FNSPID Data", fontsize=16, fontweight="bold")
    ax.set_ylabel("P&L (arbitrary units)", fontsize=12)
    ax.set_xlabel("Agent", fontsize=12)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()


def plot_agent_actions(action_df: pd.DataFrame) -> None:
    """Visualize all agent action patterns over time."""
    action_cols = [col for col in action_df.columns if col.startswith("action_")]
    if not action_cols:
        logger.warning("No action columns found to plot")
        return
    
    # Create subplots for each agent
    fig, axes = plt.subplots(len(action_cols), 1, figsize=(14, 3 * len(action_cols)), sharex=True)
    if len(action_cols) == 1:
        axes = [axes]
    
    # Map actions to numeric values for plotting
    action_map = {"buy": 1, "sell": -1, "hold": 0}
    colors = {"buy": "#55A868", "sell": "#DD8452", "hold": "#4C72B0"}
    
    for idx, col in enumerate(action_cols):
        ax = axes[idx]
        agent_name = col.replace("action_", "").replace("_", " ").title()
        
        # Convert actions to numeric
        actions_numeric = action_df[col].map(action_map)
        
        # Create color list for each point
        point_colors = [colors.get(action, "#808080") for action in action_df[col]]
        
        # Plot with colored markers
        ax.scatter(action_df["date"], actions_numeric, c=point_colors, s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.plot(action_df["date"], actions_numeric, linewidth=0.8, alpha=0.4, color='gray')
        
        ax.set_ylabel(agent_name, rotation=0, ha="right", va="center", fontsize=11, fontweight='bold')
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["SELL", "HOLD", "BUY"])
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        
        # Add background shading
        ax.axhspan(-1.5, -0.5, alpha=0.05, color='red')
        ax.axhspan(0.5, 1.5, alpha=0.05, color='green')
        ax.set_ylim(-1.5, 1.5)
    
    axes[-1].set_xlabel("Date", fontsize=12, fontweight='bold')
    fig.suptitle("Agent Trading Decisions Over Time (FNSPID Data)", y=0.995, fontsize=16, fontweight='bold')
    fig.tight_layout()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors["buy"], label='Buy'),
        Patch(facecolor=colors["hold"], label='Hold'),
        Patch(facecolor=colors["sell"], label='Sell')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=10)


def main():
    parser = argparse.ArgumentParser(description="Run LNES experiment with FNSPID data")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory")
    parser.add_argument("--backend", choices=["sentence-transformer", "tfidf"], default="tfidf",
                       help="Embedding backend")
    parser.add_argument("--num-clusters", type=int, default=5, help="Number of news clusters")
    parser.add_argument("--alpha", type=float, default=0.05, help="Price impact parameter")
    parser.add_argument("--noise", type=float, default=0.5, help="Noise standard deviation")
    parser.add_argument("--save-actions", type=Path, default=None, help="Save actions to CSV")
    
    args = parser.parse_args()
    
    if args.data_dir is None:
        args.data_dir = Path(__file__).resolve().parents[1] / "data"
    
    logger.info("=" * 70)
    logger.info(f"FNSPID EXPERIMENT - {args.ticker}")
    logger.info("=" * 70)
    
    # Load processed FNSPID data
    news_file = args.data_dir / f"fnspid_{args.ticker}_news.csv"
    price_file = args.data_dir / f"fnspid_{args.ticker}_prices.csv"
    
    if not news_file.exists():
        logger.error(f"News file not found: {news_file}")
        logger.info(f"Run: python scripts/process_fnspid_local.py --ticker {args.ticker}")
        return 1
    
    if not price_file.exists():
        logger.error(f"Price file not found: {price_file}")
        logger.info(f"Run: python scripts/process_fnspid_local.py --ticker {args.ticker}")
        return 1
    
    logger.info(f"Loading FNSPID data from {args.data_dir}")
    news = preprocess_news(load_news(news_file))
    prices = load_prices(price_file)
    merged = merge_news_and_prices(news, prices)
    logger.info(f"Loaded {len(merged)} days of market data")
    
    if len(merged) < 10:
        logger.error(f"Insufficient data: only {len(merged)} days")
        logger.info("Try a longer date range when processing FNSPID data")
        return 1
    
    # Embed and cluster news
    logger.info(f"Embedding news with {args.backend} backend...")
    embedder = NewsEmbedder(backend=args.backend)
    embeddings = embedder.encode_news(merged["combined_text"])
    
    logger.info(f"Clustering into {args.num_clusters} groups...")
    cluster_result = cluster_embeddings(embeddings, k=args.num_clusters)
    merged["cluster_id"] = cluster_result.labels
    
    # Calculate cluster sentiment
    cluster_sentiment = merged.groupby("cluster_id")["sentiment_hint"].agg(
        lambda col: col.eq("positive").mean() - col.eq("negative").mean()
    ).to_dict()
    
    logger.info(f"Cluster sentiment scores: {cluster_sentiment}")
    
    # Create agents
    agents = [
        RandomAgent(name="random"),
        MomentumAgent(name="momentum"),
        ContrarianAgent(name="contrarian"),
        NewsReactiveAgent(name="news_cluster", cluster_sentiment=cluster_sentiment),
    ]
    
    logger.info(f"Running simulation with {len(agents)} agents...")
    logger.info(f"Parameters: alpha={args.alpha}, noise={args.noise}")
    
    # Run simulation
    sim_result = simulate_market(merged, agents, alpha=args.alpha, noise_std=args.noise)
    
    # Calculate metrics
    metrics = summarize_metrics(
        reference_prices=merged["close"].tolist(),
        simulated_prices=sim_result.prices,
        action_log=sim_result.action_log,
        clusters=merged["cluster_id"].tolist(),
    )
    
    # Display results
    print("\n" + "=" * 70)
    print(f"SIMULATION RESULTS - FNSPID Dataset ({args.ticker})")
    print("=" * 70)
    print(f"Dataset Size:              {len(merged)} trading days")
    print(f"Directional Accuracy:      {metrics['directional_accuracy']:.2%}")
    print(f"Volatility Clustering:     {metrics['volatility_clustering']:.4f}")
    print(f"Cluster-Price Correlation: {metrics['cluster_price_correlation']:.4f}")
    
    print("\nAgent Profitability:")
    print(metrics["agent_profitability"].to_string(index=False))
    
    print("\nDecision Correlation Matrix:")
    print(metrics["decision_correlation"].to_string())
    
    # Save actions if requested
    if args.save_actions:
        sim_result.to_frame().to_csv(args.save_actions, index=False)
        logger.info(f"\nActions saved to: {args.save_actions}")
    else:
        # Save with default name
        default_output = f"fnspid_{args.ticker}_actions.csv"
        sim_result.to_frame().to_csv(default_output, index=False)
        logger.info(f"\nActions saved to: {default_output}")
    
    # Plot results
    logger.info("\nGenerating visualizations...")
    sim_frame = sim_result.to_frame()
    
    # Price comparison plot
    simulated_prices = sim_frame.set_index("date")["simulated_close"]
    reference_prices = merged.set_index("date")["close"]
    plot_prices(reference=reference_prices, simulated=simulated_prices, ticker=args.ticker)
    
    # Cluster distribution
    plot_cluster_distribution(cluster_result.labels)
    
    # Agent profitability
    plot_agent_profitability(metrics["agent_profitability"])
    
    # Agent actions over time - NEW
    logger.info("Plotting agent decisions over time...")
    plot_agent_actions(sim_frame)
    
    plt.show()
    
    print("\n" + "=" * 70)
    logger.info("FNSPID Experiment Complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())

