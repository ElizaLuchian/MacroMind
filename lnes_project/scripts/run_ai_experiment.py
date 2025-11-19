"""Example: Running experiments with FREE AI agents (FinBERT + Groq)."""

import argparse
import logging
from pathlib import Path

from src.agents import (
    ContrarianAgent,
    FinBERTAgent,
    GroqAgent,
    MomentumAgent,
    NewsReactiveAgent,
    RandomAgent,
)
from src.clusterer import cluster_embeddings, plot_cluster_distribution
from src.data_loader import load_news, load_prices, merge_news_and_prices
from src.embedder import NewsEmbedder
from src.metrics import summarize_metrics
from src.preprocess import preprocess_news
from src.simulator import simulate_market

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_ai_experiment(
    use_finbert: bool = True,
    use_groq: bool = False,
    groq_api_key: str | None = None,
):
    """Run experiment with AI agents."""
    logger.info("🚀 Starting AI Agent Experiment")

    # Load data
    data_dir = Path(__file__).resolve().parents[1] / "data"
    news = preprocess_news(load_news(data_dir / "small_news.csv"))
    prices = load_prices(data_dir / "small_prices.csv")
    merged = merge_news_and_prices(news, prices)
    logger.info(f"📊 Loaded {len(merged)} days of data")

    # Embed and cluster
    embedder = NewsEmbedder(backend="tfidf")  # Fast for testing
    embeddings = embedder.encode_news(merged["combined_text"])
    cluster_result = cluster_embeddings(embeddings, k=3)
    merged["cluster_id"] = cluster_result.labels
    logger.info(f"🎯 Created {len(set(cluster_result.labels))} news clusters")

    # Calculate cluster sentiment
    cluster_sentiment = merged.groupby("cluster_id")["sentiment_hint"].agg(
        lambda col: col.eq("positive").mean() - col.eq("negative").mean()
    )

    # Build agent list
    agents = [
        RandomAgent(name="random"),
        MomentumAgent(name="momentum"),
        ContrarianAgent(name="contrarian"),
        NewsReactiveAgent(name="news_cluster", cluster_sentiment=cluster_sentiment.to_dict()),
    ]

    # Add AI agents
    if use_finbert:
        try:
            logger.info("🤖 Initializing FinBERT agent (FREE, runs locally)...")
            finbert = FinBERTAgent(name="finbert_ai", confidence_threshold=0.7)
            agents.append(finbert)
            logger.info("✅ FinBERT agent ready!")
        except Exception as e:
            logger.error(f"❌ FinBERT failed: {e}")
            logger.info("Install with: pip install transformers torch")

    if use_groq:
        try:
            logger.info("🤖 Initializing Groq LLM agent (FREE tier available)...")
            groq = GroqAgent(name="groq_llama", api_key=groq_api_key)
            agents.append(groq)
            logger.info("✅ Groq agent ready!")
        except Exception as e:
            logger.error(f"❌ Groq failed: {e}")
            logger.info("Get free API key at: https://console.groq.com/keys")
            logger.info("Set it with: export GROQ_API_KEY='your-key-here'")

    # Run simulation
    logger.info(f"🎮 Running simulation with {len(agents)} agents...")
    sim_result = simulate_market(merged, agents, alpha=0.02, noise_std=0.001)

    # Calculate metrics
    metrics = summarize_metrics(
        reference_prices=merged["close"].tolist(),
        simulated_prices=sim_result.prices,
        action_log=sim_result.action_log,
        clusters=merged["cluster_id"].tolist(),
    )

    # Display results
    print("\n" + "=" * 70)
    print("📈 SIMULATION RESULTS")
    print("=" * 70)
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    print(f"Volatility Clustering: {metrics['volatility_clustering']:.4f}")
    print(f"Cluster-Price Correlation: {metrics['cluster_price_correlation']:.4f}")

    print("\n💰 Agent Profitability:")
    print(metrics["agent_profitability"].to_string(index=False))

    print("\n🤝 Decision Correlation Matrix:")
    print(metrics["decision_correlation"])

    # Save actions
    output_file = "ai_agent_actions.csv"
    sim_result.to_frame().to_csv(output_file, index=False)
    logger.info(f"\n💾 Actions saved to: {output_file}")

    return {"simulation": sim_result, "metrics": metrics, "merged": merged}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI agent trading experiment")
    parser.add_argument(
        "--finbert", action="store_true", default=True, help="Use FinBERT agent (default: True)"
    )
    parser.add_argument(
        "--no-finbert", action="store_false", dest="finbert", help="Disable FinBERT agent"
    )
    parser.add_argument("--groq", action="store_true", help="Use Groq LLM agent (requires API key)")
    parser.add_argument("--groq-api-key", type=str, help="Groq API key (or set GROQ_API_KEY env var)")

    args = parser.parse_args()

    run_ai_experiment(
        use_finbert=args.finbert, use_groq=args.groq, groq_api_key=args.groq_api_key
    )



