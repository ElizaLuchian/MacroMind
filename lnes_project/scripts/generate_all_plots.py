"""Generate complete gallery of visualizations for an experiment.

This script runs an experiment and generates all available plots,
organizing them in an output directory with an index HTML page.

Usage:
    python scripts/generate_all_plots.py [--config CONFIG] [--output-dir DIR]
    
Examples:
    # Generate plots for small dataset
    python scripts/generate_all_plots.py --config small_dataset
    
    # Generate plots for FNSPID
    python scripts/generate_all_plots.py --config fnspid_aapl --output-dir results/fnspid_plots
    
    # Compare multiple experiments
    python scripts/generate_all_plots.py --compare --configs small_dataset fnspid_aapl
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import data_loader, preprocess, embedder, clusterer, agents, simulator, metrics
from src import visualizations as viz
from src.config_loader import load_config, list_available_configs
from src.result_cache import ResultCache
import matplotlib.pyplot as plt
import numpy as np


def run_complete_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run complete experiment and return all results.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with all results and intermediate data
    """
    print(f"\n{'='*70}")
    print(f"Running Experiment: {config['experiment']['name']}")
    print(f"{'='*70}\n")
    
    # Load data
    print("1. Loading data...")
    dataset_name = config['experiment']['dataset']
    if dataset_name == "small":
        news_df = data_loader.load_news(config['dataset']['news_file'])
        prices_df = data_loader.load_prices(config['dataset']['prices_file'])
    else:
        from src.data_loader import load_fnspid
        news_df, prices_df = load_fnspid(
            ticker=config['dataset']['fnspid']['ticker'],
            start_date=config['dataset']['fnspid']['start_date'],
            end_date=config['dataset']['fnspid']['end_date'],
        )
    print(f"   Loaded {len(news_df)} news items, {len(prices_df)} price points")
    
    # Preprocess
    print("2. Preprocessing text...")
    preprocessed_texts = preprocess.clean_and_combine_df(
        news_df,
        headline_col=config['dataset']['headline_col'],
        text_col=config['dataset']['text_col'],
    )
    
    # Embed
    print("3. Generating embeddings...")
    backend = config['embedder']['backend']
    embed_model = embedder.get_embedder(backend=backend)
    embeddings = embed_model.embed_documents(preprocessed_texts)
    print(f"   Embedding shape: {embeddings.shape}")
    
    # Cluster
    print("4. Clustering...")
    k = config['clustering']['k']
    cluster_model = clusterer.KMeansClusterer(k=k)
    cluster_labels = cluster_model.fit_predict(embeddings)
    print(f"   Found {k} clusters")
    
    # Initialize agents
    print("5. Initializing agents...")
    agent_configs = config['agents']['types']
    agent_list = []
    for agent_type in agent_configs:
        if agent_type == 'random':
            agent_list.append(agents.RandomAgent())
        elif agent_type == 'momentum':
            agent_list.append(agents.MomentumAgent())
        elif agent_type == 'contrarian':
            agent_list.append(agents.ContrarianAgent())
        elif agent_type == 'newsreactive':
            agent_list.append(agents.NewsReactiveAgent(cluster_labels))
    print(f"   Initialized {len(agent_list)} agents")
    
    # Simulate
    print("6. Running simulation...")
    sim = simulator.MarketSimulator(
        reference_prices=prices_df['close'].tolist(),
        alpha=config['simulator']['alpha'],
        noise_std=config['simulator']['noise_std'],
    )
    
    sim_prices, action_log = sim.run(agent_list)
    print(f"   Simulation complete: {len(sim_prices)} periods")
    
    # Compute metrics
    print("7. Computing metrics...")
    perf_report = metrics.generate_performance_report(
        prices=sim_prices,
        reference_prices=prices_df['close'].tolist(),
        risk_free_rate=config['metrics']['risk_free_rate'],
    )
    
    agent_metrics = metrics.compare_agents_report(action_log, sim_prices)
    
    print(f"\n{'='*70}")
    print("Experiment Complete!")
    print(f"{'='*70}\n")
    
    return {
        'news_df': news_df,
        'prices_df': prices_df,
        'preprocessed_texts': preprocessed_texts,
        'embeddings': embeddings,
        'cluster_labels': cluster_labels,
        'sim_prices': sim_prices,
        'action_log': action_log,
        'perf_report': perf_report,
        'agent_metrics': agent_metrics,
        'config': config,
    }


def generate_all_plots(results: Dict[str, Any], output_dir: str) -> List[Dict[str, str]]:
    """Generate all available plots and save to directory.
    
    Args:
        results: Experiment results
        output_dir: Directory to save plots
        
    Returns:
        List of plot metadata dictionaries
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plot_metadata = []
    
    print("\n" + "="*70)
    print("Generating Plots")
    print("="*70 + "\n")
    
    # 1. Performance plots
    print("1. Performance plots...")
    
    # Equity curve
    fig = viz.plot_equity_curve(
        results['sim_prices'],
        reference_prices=results['prices_df']['close'].tolist(),
        show_drawdown=True,
    )
    filepath = os.path.join(output_dir, "01_equity_curve")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '01_equity_curve.png',
        'title': 'Equity Curve with Drawdown',
        'category': 'Performance',
    })
    plt.close(fig)
    
    # Underwater plot
    fig = viz.plot_underwater(results['sim_prices'])
    filepath = os.path.join(output_dir, "02_underwater")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '02_underwater.png',
        'title': 'Underwater Plot',
        'category': 'Performance',
    })
    plt.close(fig)
    
    # Returns distribution
    fig = viz.plot_returns_distribution(results['sim_prices'])
    filepath = os.path.join(output_dir, "03_returns_distribution")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '03_returns_distribution.png',
        'title': 'Returns Distribution',
        'category': 'Performance',
    })
    plt.close(fig)
    
    # QQ plot
    fig = viz.plot_qq(results['sim_prices'])
    filepath = os.path.join(output_dir, "04_qq_plot")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '04_qq_plot.png',
        'title': 'Q-Q Plot',
        'category': 'Performance',
    })
    plt.close(fig)
    
    # Rolling Sharpe
    fig = viz.plot_rolling_sharpe(results['sim_prices'], window=20)
    filepath = os.path.join(output_dir, "05_rolling_sharpe")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '05_rolling_sharpe.png',
        'title': 'Rolling Sharpe Ratio',
        'category': 'Performance',
    })
    plt.close(fig)
    
    # Rolling volatility
    fig = viz.plot_rolling_volatility(results['sim_prices'], window=20)
    filepath = os.path.join(output_dir, "06_rolling_volatility")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '06_rolling_volatility.png',
        'title': 'Rolling Volatility',
        'category': 'Performance',
    })
    plt.close(fig)
    
    print("   Generated 6 performance plots")
    
    # 2. Agent analysis plots
    print("2. Agent analysis plots...")
    
    # Agent PnL comparison
    fig = viz.plot_agent_comparison(results['action_log'], results['sim_prices'], metric='pnl')
    filepath = os.path.join(output_dir, "07_agent_pnl")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '07_agent_pnl.png',
        'title': 'Agent P&L Comparison',
        'category': 'Agent Analysis',
    })
    plt.close(fig)
    
    # Win rate comparison
    fig = viz.plot_agent_comparison(results['action_log'], results['sim_prices'], metric='win_rate')
    filepath = os.path.join(output_dir, "08_agent_winrate")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '08_agent_winrate.png',
        'title': 'Agent Win Rate Comparison',
        'category': 'Agent Analysis',
    })
    plt.close(fig)
    
    # Action heatmap
    fig = viz.plot_agent_action_heatmap(results['action_log'])
    filepath = os.path.join(output_dir, "09_action_heatmap")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '09_action_heatmap.png',
        'title': 'Agent Action Heatmap',
        'category': 'Agent Analysis',
    })
    plt.close(fig)
    
    # Action frequency
    fig = viz.plot_action_frequency(results['action_log'])
    filepath = os.path.join(output_dir, "10_action_frequency")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '10_action_frequency.png',
        'title': 'Action Frequency by Agent',
        'category': 'Agent Analysis',
    })
    plt.close(fig)
    
    # Agent correlation network
    fig = viz.plot_agent_correlation_network(results['action_log'])
    filepath = os.path.join(output_dir, "11_agent_network")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '11_agent_network.png',
        'title': 'Agent Correlation Network',
        'category': 'Agent Analysis',
    })
    plt.close(fig)
    
    # Rolling PnL
    fig = viz.plot_agent_rolling_pnl(results['action_log'], results['sim_prices'], window=10)
    filepath = os.path.join(output_dir, "12_rolling_pnl")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '12_rolling_pnl.png',
        'title': 'Agent Rolling P&L',
        'category': 'Agent Analysis',
    })
    plt.close(fig)
    
    print("   Generated 6 agent analysis plots")
    
    # 3. Embedding visualization
    print("3. Embedding visualizations...")
    
    # t-SNE
    fig = viz.plot_embedding_tsne(
        results['embeddings'],
        labels=results['cluster_labels'],
        texts=results['preprocessed_texts'],
    )
    filepath = os.path.join(output_dir, "13_tsne")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '13_tsne.png',
        'title': 't-SNE Embedding Visualization',
        'category': 'Embeddings',
    })
    plt.close(fig)
    
    # UMAP
    fig = viz.plot_embedding_umap(
        results['embeddings'],
        labels=results['cluster_labels'],
        texts=results['preprocessed_texts'],
    )
    filepath = os.path.join(output_dir, "14_umap")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '14_umap.png',
        'title': 'UMAP Embedding Visualization',
        'category': 'Embeddings',
    })
    plt.close(fig)
    
    # Silhouette analysis
    fig = viz.plot_cluster_silhouette(results['embeddings'], results['cluster_labels'])
    filepath = os.path.join(output_dir, "15_silhouette")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '15_silhouette.png',
        'title': 'Cluster Silhouette Analysis',
        'category': 'Embeddings',
    })
    plt.close(fig)
    
    print("   Generated 3 embedding plots")
    
    # 4. Comprehensive dashboard
    print("4. Comprehensive dashboard...")
    
    fig = viz.plot_comprehensive_dashboard(
        results['sim_prices'],
        action_log=results['action_log'],
        reference_prices=results['prices_df']['close'].tolist(),
    )
    filepath = os.path.join(output_dir, "16_dashboard")
    viz.save_figure(fig, filepath, formats=['png', 'pdf'])
    plot_metadata.append({
        'filename': '16_dashboard.png',
        'title': 'Comprehensive Performance Dashboard',
        'category': 'Dashboard',
    })
    plt.close(fig)
    
    print("   Generated comprehensive dashboard")
    
    print(f"\nTotal plots generated: {len(plot_metadata)}")
    
    return plot_metadata


def create_index_html(plot_metadata: List[Dict[str, str]], output_dir: str, 
                     config: Dict[str, Any], perf_report: Dict[str, Any]):
    """Create HTML index page with all plots.
    
    Args:
        plot_metadata: List of plot metadata
        output_dir: Output directory
        config: Experiment configuration
        perf_report: Performance report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MacroMind Experiment Gallery</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-left: 4px solid #4CAF50;
            padding-left: 10px;
        }}
        .metadata {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #2196F3;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .plot-item {{
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .plot-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .plot-item img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .plot-title {{
            padding: 15px;
            background-color: #fafafa;
            font-weight: 600;
            color: #333;
        }}
        .category-badge {{
            display: inline-block;
            padding: 4px 8px;
            background-color: #4CAF50;
            color: white;
            border-radius: 3px;
            font-size: 11px;
            margin-left: 10px;
        }}
        .footer {{
            margin-top: 50px;
            padding: 20px;
            text-align: center;
            color: #999;
            border-top: 1px solid #ddd;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <h1>📈 MacroMind Experiment Gallery</h1>
    
    <div class="metadata">
        <h2>Experiment Details</h2>
        <p><strong>Name:</strong> {config['experiment']['name']}</p>
        <p><strong>Dataset:</strong> {config['experiment']['dataset']}</p>
        <p><strong>Embedding Backend:</strong> {config['embedder']['backend']}</p>
        <p><strong>Number of Clusters:</strong> {config['clustering']['k']}</p>
        <p><strong>Agents:</strong> {', '.join(config['agents']['types'])}</p>
        <p><strong>Generated:</strong> {timestamp}</p>
        
        <h3>Key Metrics</h3>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{perf_report.get('total_return', 0)*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{perf_report.get('sharpe_ratio', 0):.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{perf_report.get('max_drawdown', 0)*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volatility (Ann.)</div>
                <div class="metric-value">{perf_report.get('volatility_annual', 0)*100:.2f}%</div>
            </div>
        </div>
    </div>
"""
    
    # Group plots by category
    categories = {}
    for plot in plot_metadata:
        category = plot['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(plot)
    
    # Add plots by category
    for category, plots in categories.items():
        html_content += f"""
    <h2>{category}</h2>
    <div class="plot-grid">
"""
        for plot in plots:
            html_content += f"""
        <div class="plot-item">
            <a href="{plot['filename']}" target="_blank">
                <img src="{plot['filename']}" alt="{plot['title']}">
            </a>
            <div class="plot-title">
                {plot['title']}
                <span class="category-badge">{category}</span>
            </div>
        </div>
"""
        html_content += """
    </div>
"""
    
    # Footer
    html_content += f"""
    <div class="footer">
        <p>Generated by MacroMind Visualization Suite</p>
        <p>Total Plots: {len(plot_metadata)} | Categories: {len(categories)}</p>
    </div>
</body>
</html>
"""
    
    # Write to file
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Created index page: {index_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate complete visualization gallery for an experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="small_dataset",
        help="Configuration name (default: small_dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/plots",
        help="Output directory for plots (default: output/plots)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved plots (default: 300)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("MacroMind Visualization Gallery Generator")
    print(f"{'='*70}\n")
    
    # Load configuration
    print(f"Loading configuration: {args.config}")
    config = load_config(args.config)
    
    # Run experiment
    results = run_complete_experiment(config)
    
    # Generate plots
    plot_metadata = generate_all_plots(results, args.output_dir)
    
    # Create index HTML
    create_index_html(plot_metadata, args.output_dir, config, results['perf_report'])
    
    # Save metrics report
    report_text = metrics.format_performance_report(results['perf_report'])
    report_path = os.path.join(args.output_dir, "metrics_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"✅ Saved metrics report: {report_path}")
    
    # Save agent metrics
    agent_csv_path = os.path.join(args.output_dir, "agent_metrics.csv")
    results['agent_metrics'].to_csv(agent_csv_path, index=False)
    print(f"✅ Saved agent metrics: {agent_csv_path}")
    
    print(f"\n{'='*70}")
    print("Gallery Generation Complete!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Open index.html in a browser to view all plots")
    print(f"\n  file://{os.path.abspath(os.path.join(args.output_dir, 'index.html'))}")
    print()


if __name__ == "__main__":
    main()

