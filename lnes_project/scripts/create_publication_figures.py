"""Generate publication-ready figures for research papers.

This script creates high-quality, journal-ready figures with:
- LaTeX-compatible fonts
- Vector format outputs (PDF, SVG)
- Multi-panel layouts with subfigure labels
- Grayscale-friendly color schemes
- Consistent sizing for academic journals

Usage:
    python scripts/create_publication_figures.py [--config CONFIG] [--output-dir DIR]
    
Examples:
    # Generate publication figures for small dataset
    python scripts/create_publication_figures.py --config small_dataset
    
    # Custom output directory
    python scripts/create_publication_figures.py --config fnspid_aapl --output-dir paper/figures
"""

import argparse
import os
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import data_loader, preprocess, embedder, clusterer, agents, simulator, metrics
from src import visualizations as viz
from src.config_loader import load_config
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# Set publication style
def set_publication_params():
    """Set matplotlib parameters for publication quality."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        # Use colorblind-friendly colors
        'axes.prop_cycle': plt.cycler(color=[
            '#0173B2', '#DE8F05', '#029E73', '#CC78BC',
            '#CA9161', '#949494', '#ECE133', '#56B4E9'
        ])
    })


def create_figure_1_overview(results, output_dir):
    """Figure 1: Experimental overview (multi-panel).
    
    Layout:
    (A) Price trajectories (simulated vs reference)
    (B) Returns distribution
    (C) Agent profitability
    (D) Cluster distribution
    """
    fig = plt.figure(figsize=(7.5, 8))  # Two-column width
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35, figure=fig)
    
    # (A) Price trajectories
    ax1 = fig.add_subplot(gs[0, :])
    sim_prices = results['sim_prices']
    ref_prices = results['prices_df']['close'].tolist()
    x = np.arange(len(sim_prices))
    
    ax1.plot(x, sim_prices, label='Simulated', linewidth=1.5, color='#0173B2')
    ax1.plot(x, ref_prices[:len(sim_prices)], label='Reference', 
            linewidth=1.5, linestyle='--', color='#DE8F05', alpha=0.8)
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Price')
    ax1.set_title('(A) Price Trajectories', fontweight='bold', loc='left')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # (B) Returns distribution
    ax2 = fig.add_subplot(gs[1, 0])
    returns = metrics.compute_returns(sim_prices)
    if len(returns) > 0:
        ax2.hist(returns, bins=30, density=True, alpha=0.7, 
                color='#029E73', edgecolor='black', linewidth=0.5)
        
        # Overlay normal distribution
        mu, sigma = np.mean(returns), np.std(returns)
        x_norm = np.linspace(min(returns), max(returns), 100)
        from scipy import stats
        ax2.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 
                'r-', linewidth=2, label='Normal')
        
        ax2.axvline(mu, color='black', linestyle='--', linewidth=1, 
                   alpha=0.7, label=f'μ={mu:.4f}')
    
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Density')
    ax2.set_title('(B) Returns Distribution', fontweight='bold', loc='left')
    ax2.legend(loc='best', framealpha=0.9, fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # (C) Agent profitability
    ax3 = fig.add_subplot(gs[1, 1])
    prof_df = metrics.agent_profitability(results['action_log'], sim_prices)
    prof_df = prof_df.sort_values('pnl', ascending=False)
    
    colors = ['#0173B2' if v > 0 else '#DE8F05' for v in prof_df['pnl']]
    bars = ax3.bar(range(len(prof_df)), prof_df['pnl'], color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_xticks(range(len(prof_df)))
    ax3.set_xticklabels(prof_df['agent'], rotation=45, ha='right')
    ax3.set_ylabel('P&L')
    ax3.set_title('(C) Agent Profitability', fontweight='bold', loc='left')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Save figure
    filepath = os.path.join(output_dir, "figure_1_overview")
    viz.save_figure(fig, filepath, formats=['pdf', 'png', 'svg'])
    plt.close(fig)
    
    print("✅ Figure 1: Experimental Overview")


def create_figure_2_performance(results, output_dir):
    """Figure 2: Performance metrics (multi-panel).
    
    Layout:
    (A) Equity curve with drawdown shading
    (B) Underwater plot
    (C) Rolling Sharpe ratio
    (D) Rolling volatility
    """
    fig = plt.figure(figsize=(7.5, 8))
    gs = gridspec.GridSpec(4, 1, hspace=0.4, figure=fig)
    
    sim_prices = results['sim_prices']
    ref_prices = results['prices_df']['close'].tolist()
    
    # (A) Equity curve
    ax1 = fig.add_subplot(gs[0])
    data = metrics.prepare_equity_curve_data(sim_prices)
    ax1.plot(data['indices'], data['prices'], label='Portfolio', 
            linewidth=1.5, color='#0173B2')
    ax1.plot(data['indices'], ref_prices[:len(sim_prices)], 
            label='Reference', linewidth=1.5, linestyle='--', 
            color='#DE8F05', alpha=0.7)
    ax1.fill_between(data['indices'], data['cummax'], data['prices'],
                     alpha=0.3, color='#DE8F05', label='Drawdown')
    ax1.set_ylabel('Price')
    ax1.set_title('(A) Equity Curve', fontweight='bold', loc='left')
    ax1.legend(loc='best', framealpha=0.9, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # (B) Underwater plot
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(data['indices'], 0, data['drawdowns'] * 100,
                     color='#DE8F05', alpha=0.7)
    ax2.plot(data['indices'], data['drawdowns'] * 100, 
            color='#CC5500', linewidth=1)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('(B) Underwater Plot', fontweight='bold', loc='left')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3)
    
    # (C) Rolling Sharpe
    ax3 = fig.add_subplot(gs[2])
    rolling_data = metrics.prepare_rolling_metric_data(sim_prices, 20, 'sharpe', 0.02)
    if len(rolling_data['values']) > 0:
        ax3.plot(rolling_data['indices'], rolling_data['values'], 
                linewidth=1.5, color='#029E73')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax3.axhline(y=1, color='#0173B2', linestyle=':', linewidth=1, 
                   alpha=0.7, label='Sharpe=1')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('(C) Rolling Sharpe Ratio (Window=20)', 
                 fontweight='bold', loc='left')
    ax3.legend(loc='best', framealpha=0.9, fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # (D) Rolling volatility
    ax4 = fig.add_subplot(gs[3])
    vol_data = metrics.prepare_rolling_metric_data(sim_prices, 20, 'volatility')
    if len(vol_data['values']) > 0:
        ax4.plot(vol_data['indices'], vol_data['values'] * 100, 
                linewidth=1.5, color='#CC78BC')
    ax4.set_xlabel('Time Period')
    ax4.set_ylabel('Annualized Vol. (%)')
    ax4.set_title('(D) Rolling Volatility (Window=20)', 
                 fontweight='bold', loc='left')
    ax4.grid(True, alpha=0.3)
    
    # Save figure
    filepath = os.path.join(output_dir, "figure_2_performance")
    viz.save_figure(fig, filepath, formats=['pdf', 'png', 'svg'])
    plt.close(fig)
    
    print("✅ Figure 2: Performance Metrics")


def create_figure_3_agents(results, output_dir):
    """Figure 3: Agent behavior analysis.
    
    Layout:
    (A) Agent action heatmap
    (B) Action frequency stacked bars
    (C) Agent correlation network (or rolling PnL)
    """
    fig = plt.figure(figsize=(7.5, 8))
    gs = gridspec.GridSpec(3, 1, hspace=0.4, figure=fig, height_ratios=[1.2, 1, 1])
    
    action_log = results['action_log']
    sim_prices = results['sim_prices']
    
    # (A) Action heatmap
    ax1 = fig.add_subplot(gs[0])
    
    # Convert actions to numeric
    action_to_num = {"buy": 1, "hold": 0, "sell": -1}
    agents = sorted(action_log.keys())
    n_periods = len(action_log[agents[0]])
    
    matrix = np.zeros((len(agents), n_periods))
    for i, agent in enumerate(agents):
        for j, action in enumerate(action_log[agent]):
            matrix[i, j] = action_to_num.get(action, 0)
    
    im = ax1.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax1.set_yticks(np.arange(len(agents)))
    ax1.set_yticklabels(agents, fontsize=9)
    ax1.set_xlabel('Time Period')
    ax1.set_title('(A) Agent Action Heatmap', fontweight='bold', loc='left')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Sell', 'Hold', 'Buy'])
    
    # (B) Action frequency
    ax2 = fig.add_subplot(gs[1])
    
    actions = ["buy", "hold", "sell"]
    data = []
    for agent in agents:
        counts = [action_log[agent].count(a) for a in actions]
        data.append(counts)
    
    import pandas as pd
    df = pd.DataFrame(data, index=agents, columns=actions)
    
    # Stacked bar
    x = np.arange(len(agents))
    width = 0.6
    
    bottom = np.zeros(len(agents))
    colors_stack = ['#029E73', '#949494', '#DE8F05']
    
    for i, action in enumerate(actions):
        ax2.bar(x, df[action], width, label=action.capitalize(), 
               bottom=bottom, color=colors_stack[i], alpha=0.8, 
               edgecolor='black', linewidth=0.5)
        bottom += df[action]
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(agents, rotation=45, ha='right')
    ax2.set_ylabel('Number of Actions')
    ax2.set_title('(B) Action Frequency by Agent', fontweight='bold', loc='left')
    ax2.legend(loc='best', framealpha=0.9, fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # (C) Rolling PnL
    ax3 = fig.add_subplot(gs[2])
    
    colors_pnl = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#949494']
    
    for i, agent in enumerate(agents):
        # Compute cumulative PnL
        pnl = []
        position = 0
        cumulative_pnl = 0
        
        for j, action in enumerate(action_log[agent]):
            if j == 0:
                pnl.append(0)
                continue
            
            price_change = sim_prices[j] - sim_prices[j - 1]
            cumulative_pnl += position * price_change
            
            if action == "buy":
                position += 1
            elif action == "sell":
                position -= 1
            
            pnl.append(cumulative_pnl)
        
        ax3.plot(pnl, label=agent, linewidth=1.5, 
                color=colors_pnl[i % len(colors_pnl)], alpha=0.8)
    
    ax3.set_xlabel('Time Period')
    ax3.set_ylabel('Cumulative P&L')
    ax3.set_title('(C) Agent Cumulative P&L', fontweight='bold', loc='left')
    ax3.legend(loc='best', framealpha=0.9, fontsize=7, ncol=2)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Save figure
    filepath = os.path.join(output_dir, "figure_3_agents")
    viz.save_figure(fig, filepath, formats=['pdf', 'png', 'svg'])
    plt.close(fig)
    
    print("✅ Figure 3: Agent Behavior Analysis")


def create_figure_4_embeddings(results, output_dir):
    """Figure 4: Embedding space visualization.
    
    Layout:
    (A) t-SNE projection with cluster colors
    (B) Silhouette analysis
    """
    fig = plt.figure(figsize=(7.5, 5))
    gs = gridspec.GridSpec(1, 2, hspace=0.3, wspace=0.4, figure=fig)
    
    embeddings = results['embeddings']
    labels = results['cluster_labels']
    
    # (A) t-SNE
    ax1 = fig.add_subplot(gs[0, 0])
    
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1),
                   random_state=42, n_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[colors[i]], label=f"C{label}", alpha=0.7, s=80, 
                       edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.set_title('(A) t-SNE Embedding Projection', 
                     fontweight='bold', loc='left')
        ax1.legend(loc='best', framealpha=0.9, fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
    except ImportError:
        ax1.text(0.5, 0.5, 't-SNE not available', ha='center', va='center')
    
    # (B) Silhouette
    ax2 = fig.add_subplot(gs[0, 1])
    
    try:
        from sklearn.metrics import silhouette_samples, silhouette_score
        
        silhouette_avg = silhouette_score(embeddings, labels)
        sample_silhouette_values = silhouette_samples(embeddings, labels)
        
        y_lower = 10
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            cluster_silhouette_values = sample_silhouette_values[labels == label]
            cluster_silhouette_values.sort()
            
            size_cluster = len(cluster_silhouette_values)
            y_upper = y_lower + size_cluster
            
            ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, 
                             cluster_silhouette_values,
                             facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
            
            ax2.text(-0.05, y_lower + 0.5 * size_cluster, f'C{label}', fontsize=8)
            
            y_lower = y_upper + 10
        
        ax2.set_xlabel('Silhouette Coefficient')
        ax2.set_ylabel('Cluster')
        ax2.set_title(f'(B) Silhouette Analysis (avg={silhouette_avg:.3f})', 
                     fontweight='bold', loc='left')
        ax2.axvline(x=silhouette_avg, color='red', linestyle='--', 
                   linewidth=1.5, label=f'Avg: {silhouette_avg:.3f}')
        ax2.legend(loc='best', framealpha=0.9, fontsize=8)
        ax2.grid(True, alpha=0.3, axis='x')
    except ImportError:
        ax2.text(0.5, 0.5, 'scikit-learn not available', ha='center', va='center')
    
    # Save figure
    filepath = os.path.join(output_dir, "figure_4_embeddings")
    viz.save_figure(fig, filepath, formats=['pdf', 'png', 'svg'])
    plt.close(fig)
    
    print("✅ Figure 4: Embedding Space Visualization")


def create_supplementary_figure_metrics_table(results, output_dir):
    """Supplementary: Detailed metrics table (as image)."""
    
    # Generate performance report
    report = results['perf_report']
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.axis('off')
    
    # Prepare data
    metrics_data = [
        ['Metric', 'Value'],
        ['', ''],  # Separator
        ['RETURN METRICS', ''],
        ['Total Return', f"{report.get('total_return', 0)*100:.2f}%"],
        ['Mean Return', f"{report.get('mean_return', 0):.6f}"],
        ['Std Deviation', f"{report.get('std_return', 0):.6f}"],
        ['Skewness', f"{report.get('skewness', 0):.4f}"],
        ['Kurtosis', f"{report.get('kurtosis', 0):.4f}"],
        ['', ''],
        ['RISK-ADJUSTED RETURNS', ''],
        ['Sharpe Ratio', f"{report.get('sharpe_ratio', 0):.4f}"],
        ['Sortino Ratio', f"{report.get('sortino_ratio', 0):.4f}"],
        ['Calmar Ratio', f"{report.get('calmar_ratio', 0):.4f}"],
        ['', ''],
        ['DRAWDOWN ANALYSIS', ''],
        ['Max Drawdown', f"{report.get('max_drawdown', 0)*100:.2f}%"],
        ['Avg Drawdown', f"{report.get('avg_drawdown', 0)*100:.2f}%"],
        ['Max DD Duration', f"{report.get('max_dd_duration', 0)} periods"],
        ['', ''],
        ['RISK METRICS', ''],
        ['VaR (95%)', f"{report.get('var_95', 0)*100:.2f}%"],
        ['CVaR (95%)', f"{report.get('cvar_95', 0)*100:.2f}%"],
        ['Volatility (Ann.)', f"{report.get('volatility_annual', 0)*100:.2f}%"],
    ]
    
    # Create table
    table = ax.table(cellText=metrics_data, cellLoc='left', loc='center',
                    colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in [0, 2, 9, 13, 17]:
        for j in range(2):
            cell = table[(i, j)]
            cell.set_facecolor('#E0E0E0')
            cell.set_text_props(weight='bold')
    
    ax.set_title('Supplementary Table 1: Performance Metrics Summary',
                fontweight='bold', fontsize=12, pad=20)
    
    # Save
    filepath = os.path.join(output_dir, "supplementary_table_1_metrics")
    viz.save_figure(fig, filepath, formats=['pdf', 'png'])
    plt.close(fig)
    
    print("✅ Supplementary Table 1: Metrics Summary")


def run_experiment_quick(config):
    """Run experiment (same as generate_all_plots but simplified)."""
    print(f"\nRunning experiment: {config['experiment']['name']}")
    
    # Load data
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
    
    # Preprocess
    preprocessed_texts = preprocess.clean_and_combine_df(
        news_df,
        headline_col=config['dataset']['headline_col'],
        text_col=config['dataset']['text_col'],
    )
    
    # Embed
    backend = config['embedder']['backend']
    embed_model = embedder.get_embedder(backend=backend)
    embeddings = embed_model.embed_documents(preprocessed_texts)
    
    # Cluster
    k = config['clustering']['k']
    cluster_model = clusterer.KMeansClusterer(k=k)
    cluster_labels = cluster_model.fit_predict(embeddings)
    
    # Initialize agents
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
    
    # Simulate
    sim = simulator.MarketSimulator(
        reference_prices=prices_df['close'].tolist(),
        alpha=config['simulator']['alpha'],
        noise_std=config['simulator']['noise_std'],
    )
    
    sim_prices, action_log = sim.run(agent_list)
    
    # Compute metrics
    perf_report = metrics.generate_performance_report(
        prices=sim_prices,
        reference_prices=prices_df['close'].tolist(),
        risk_free_rate=config['metrics']['risk_free_rate'],
    )
    
    return {
        'news_df': news_df,
        'prices_df': prices_df,
        'preprocessed_texts': preprocessed_texts,
        'embeddings': embeddings,
        'cluster_labels': cluster_labels,
        'sim_prices': sim_prices,
        'action_log': action_log,
        'perf_report': perf_report,
        'config': config,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for research papers"
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
        default="output/publication_figures",
        help="Output directory (default: output/publication_figures)"
    )
    
    args = parser.parse_args()
    
    # Set publication style
    set_publication_params()
    
    print(f"\n{'='*70}")
    print("MacroMind Publication Figure Generator")
    print(f"{'='*70}\n")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    print(f"Loading configuration: {args.config}")
    config = load_config(args.config)
    
    # Run experiment
    results = run_experiment_quick(config)
    
    # Generate figures
    print(f"\nGenerating publication figures...")
    print(f"Output directory: {args.output_dir}\n")
    
    create_figure_1_overview(results, args.output_dir)
    create_figure_2_performance(results, args.output_dir)
    create_figure_3_agents(results, args.output_dir)
    create_figure_4_embeddings(results, args.output_dir)
    create_supplementary_figure_metrics_table(results, args.output_dir)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✅ Saved configuration: {config_path}")
    
    print(f"\n{'='*70}")
    print("Publication Figures Complete!")
    print(f"{'='*70}")
    print(f"\nGenerated 4 main figures + 1 supplementary table")
    print(f"All figures saved in: {args.output_dir}")
    print(f"\nFormats: PDF (vector), PNG (raster), SVG (web)")
    print()


if __name__ == "__main__":
    main()

