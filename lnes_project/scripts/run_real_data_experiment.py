"""Run experiment with real market data downloaded from Yahoo Finance."""

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


def plot_prices(reference: pd.Series, simulated: pd.Series) -> None:
    """Plot reference vs simulated prices."""
    fig, ax = plt.subplots(figsize=(12, 6))
    reference = reference.sort_index()
    simulated = simulated.sort_index()
    ax.plot(reference.index, reference.values, label="Real Market Price", color="#4C72B0", linewidth=2)
    ax.plot(simulated.index, simulated.values, label="Simulated Price", color="#DD8452", linestyle="--", linewidth=2)
    ax.set_title("Real Market vs Simulated Prices", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def plot_agent_profitability(df: pd.DataFrame) -> None:
    """Plot agent profitability."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["agent"], df["pnl"], color="#55A868")
    ax.set_title("Agent Profitability (P&L)", fontsize=14, fontweight="bold")
    ax.set_ylabel("P&L (arbitrary units)")
    ax.set_xlabel("Agent")
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Run LNES experiment with real market data")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker (default: AAPL)")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory")
    parser.add_argument("--backend", choices=["sentence-transformer", "tfidf"], default="tfidf",
                       help="Embedding backend (default: tfidf for speed)")
    parser.add_argument("--num-clusters", type=int, default=5, help="Number of news clusters (default: 5)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Price impact parameter (default: 0.05)")
    parser.add_argument("--noise", type=float, default=0.5, help="Noise standard deviation (default: 0.5)")
    parser.add_argument("--save-actions", type=Path, default=None, help="Save actions to CSV file")
    
    args = parser.parse_args()
    
    if args.data_dir is None:
        args.data_dir = Path(__file__).resolve().parents[1] / "data"
    
    logger.info(f"Running experiment with {args.ticker} data")
    logger.info("=" * 70)
    
    # Load data
    news_file = args.data_dir / f"{args.ticker}_news.csv"
    price_file = args.data_dir / f"{args.ticker}_prices.csv"
    
    if not news_file.exists():
        logger.error(f"News file not found: {news_file}")
        logger.info(f"Run: python scripts/download_real_data.py --ticker {args.ticker}")
        return
    
    if not price_file.exists():
        logger.error(f"Price file not found: {price_file}")
        logger.info(f"Run: python scripts/download_real_data.py --ticker {args.ticker}")
        return
    
    logger.info(f"Loading data from {args.data_dir}")
    news = preprocess_news(load_news(news_file))
    prices = load_prices(price_file)
    merged = merge_news_and_prices(news, prices)
    logger.info(f"Loaded {len(merged)} days of market data")
    
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
    print(f"SIMULATION RESULTS - {args.ticker}")
    print("=" * 70)
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
    
    # Plot results
    logger.info("\nGenerating visualizations...")
    sim_frame = sim_result.to_frame().set_index("date")
    simulated_prices = sim_frame["simulated_close"]
    reference_prices = merged.set_index("date")["close"]
    
    plot_prices(reference=reference_prices, simulated=simulated_prices)
    plot_cluster_distribution(cluster_result.labels)
    plot_agent_profitability(metrics["agent_profitability"])
    
    plt.show()
    
    print("\n" + "=" * 70)
    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()

