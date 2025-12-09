"""
Notebook Utilities for MacroMind (LNES) Analysis

This module provides helper functions for Jupyter notebooks including
common imports, plotting configuration, experiment runners, and result
comparison utilities.

Author: MacroMind Research Team
Date: December 2024
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == 'notebooks' else notebook_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple

# MacroMind imports
from src.config_loader import load_config, ConfigLoader
from src.result_cache import ResultCache, save_results, load_results
from src.data_loader import load_smallset, load_fnspid
from src.embedder import NewsEmbedder
from src.clusterer import NewsClustering
from src.agents import Random, Momentum, Contrarian, NewsReactive
from src.simulator import MarketSimulator
from src import metrics
from src.visualizations import (
    plot_equity_curve,
    plot_underwater,
    plot_returns_distribution,
    plot_rolling_sharpe,
    plot_comprehensive_dashboard
)

# Plotting style configuration
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGSIZE_DEFAULT = (12, 8)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 10)
DPI = 100

# Color palette for agents
AGENT_COLORS = {
    'Random': '#95a5a6',
    'Momentum': '#3498db',
    'Contrarian': '#e74c3c',
    'NewsReactive': '#2ecc71',
    'FinBERT': '#9b59b6',
    'Groq': '#f39c12',
}


def setup_plotting_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use(PLOT_STYLE)
    plt.rcParams['figure.figsize'] = FIGSIZE_DEFAULT
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def print_subsection(title: str) -> None:
    """Print formatted subsection header."""
    print(f"\n{title}")
    print("-" * len(title))


def quick_experiment(config_name: str = "small_dataset", 
                    verbose: bool = True) -> Dict[str, Any]:
    """
    Run a complete experiment quickly using configuration.
    
    Parameters
    ----------
    config_name : str
        Name of configuration file (without .yaml)
    verbose : bool
        Print progress messages
    
    Returns
    -------
    results : dict
        Complete experiment results including:
        - config: Configuration used
        - news_df: News DataFrame
        - prices_df: Prices DataFrame
        - embeddings: News embeddings
        - clusters: Cluster assignments
        - cluster_model: Fitted clustering model
        - agents: Agent instances
        - action_log: Agent actions
        - sim_prices: Simulated prices
        - ref_prices: Reference prices
        - metrics: Performance metrics
    """
    if verbose:
        print_section(f"Running Experiment: {config_name}")
    
    # Load configuration
    if verbose:
        print("Loading configuration...")
    config = load_config(config_name)
    
    # Load data
    if verbose:
        print("Loading data...")
    dataset_type = config['dataset']['type']
    
    if dataset_type == 'small':
        news_df, prices_df = load_smallset()
    elif dataset_type == 'fnspid':
        tickers = config['dataset']['fnspid']['tickers']
        start_date = config['dataset']['fnspid']['start_date']
        end_date = config['dataset']['fnspid']['end_date']
        news_df, prices_df = load_fnspid(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    if verbose:
        print(f"  Loaded {len(news_df)} news items and {len(prices_df)} price records")
    
    # Embed news
    if verbose:
        print("Embedding news...")
    embedder_config = config['embedder']
    embedder = NewsEmbedder(backend=embedder_config['backend'])
    embeddings = embedder.embed(news_df['text'].tolist())
    
    if verbose:
        print(f"  Generated embeddings of shape {embeddings.shape}")
    
    # Cluster news
    if verbose:
        print("Clustering news...")
    k = config['clustering']['k']
    clustering = NewsClustering(k=k)
    clusters = clustering.fit_predict(embeddings)
    
    if verbose:
        print(f"  Created {k} clusters")
        print(f"  Silhouette score: {clustering.silhouette_score(embeddings):.3f}")
    
    # Initialize agents
    if verbose:
        print("Initializing agents...")
    agents_list = []
    agent_names = config['agents']['enabled']
    
    for agent_name in agent_names:
        if agent_name == 'Random':
            agents_list.append(Random())
        elif agent_name == 'Momentum':
            agents_list.append(Momentum())
        elif agent_name == 'Contrarian':
            agents_list.append(Contrarian())
        elif agent_name == 'NewsReactive':
            agents_list.append(NewsReactive(clusters=clusters, k=k))
        # Note: FinBERT and Groq require additional setup
    
    if verbose:
        print(f"  Initialized {len(agents_list)} agents: {agent_names}")
    
    # Run simulation
    if verbose:
        print("Running market simulation...")
    simulator = MarketSimulator(
        agents=agents_list,
        alpha=config['simulator']['alpha'],
        noise_std=config['simulator']['noise_std']
    )
    
    sim_prices, action_log = simulator.simulate(
        news_df=news_df,
        prices_df=prices_df,
        clusters=clusters
    )
    
    if verbose:
        print(f"  Simulated {len(sim_prices)} time steps")
    
    # Compute metrics
    if verbose:
        print("Computing metrics...")
    ref_prices = prices_df['close'].values
    
    dir_acc = metrics.directional_accuracy(ref_prices, sim_prices)
    vol_clust = metrics.volatility_clustering(sim_prices)
    
    metrics_dict = {
        'directional_accuracy': dir_acc,
        'volatility_clustering': vol_clust,
    }
    
    if config['metrics']['advanced_metrics']:
        sharpe = metrics.sharpe_ratio(sim_prices, 
                                     risk_free_rate=config['metrics']['risk_free_rate'])
        max_dd = metrics.maximum_drawdown(sim_prices)
        metrics_dict['sharpe_ratio'] = sharpe
        metrics_dict['max_drawdown'] = max_dd
    
    if verbose:
        print(f"  Directional accuracy: {dir_acc:.2%}")
        print(f"  Volatility clustering: {vol_clust:.3f}")
    
    # Compile results
    results = {
        'config': config,
        'news_df': news_df,
        'prices_df': prices_df,
        'embeddings': embeddings,
        'clusters': clusters,
        'cluster_model': clustering,
        'agents': agents_list,
        'action_log': action_log,
        'sim_prices': sim_prices,
        'ref_prices': ref_prices,
        'metrics': metrics_dict,
    }
    
    if verbose:
        print_section("Experiment Complete!")
    
    return results


def compare_experiments(cache_keys: List[str], 
                       metric: str = 'directional_accuracy') -> pd.DataFrame:
    """
    Compare multiple experiments by a specific metric.
    
    Parameters
    ----------
    cache_keys : list of str
        List of cache keys to compare
    metric : str
        Metric to compare
    
    Returns
    -------
    comparison : pd.DataFrame
        Comparison table
    """
    cache = ResultCache()
    
    data = []
    for key in cache_keys:
        try:
            results, config = cache.load(key)
            metric_value = results.get('metrics', {}).get(metric, None)
            
            data.append({
                'cache_key': key,
                'experiment': config['experiment']['name'],
                'dataset': config['dataset']['type'],
                'agents': ', '.join(config['agents']['enabled']),
                metric: metric_value,
            })
        except Exception as e:
            print(f"Warning: Could not load {key}: {e}")
    
    return pd.DataFrame(data)


def plot_agent_comparison(action_log: Dict[str, List[str]], 
                         prices: np.ndarray,
                         figsize: Tuple[int, int] = FIGSIZE_DEFAULT) -> plt.Figure:
    """
    Plot agent performance comparison.
    
    Parameters
    ----------
    action_log : dict
        Agent action log
    prices : np.ndarray
        Price series
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Calculate metrics
    pnl = metrics.agent_pnl(action_log, prices)
    win_rates = metrics.win_rate(action_log, prices)
    dir_acc = metrics.per_agent_directional_accuracy(action_log, prices)
    
    agent_names = list(action_log.keys())
    
    # PnL
    axes[0, 0].bar(agent_names, [pnl[a] for a in agent_names],
                   color=[AGENT_COLORS.get(a, '#95a5a6') for a in agent_names])
    axes[0, 0].set_title('Agent PnL (Naive)')
    axes[0, 0].set_ylabel('Total PnL')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Win Rates
    axes[0, 1].bar(agent_names, [win_rates[a] for a in agent_names],
                   color=[AGENT_COLORS.get(a, '#95a5a6') for a in agent_names])
    axes[0, 1].set_title('Win Rate')
    axes[0, 1].set_ylabel('Win Rate')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    
    # Directional Accuracy
    axes[1, 0].bar(agent_names, [dir_acc[a] for a in agent_names],
                   color=[AGENT_COLORS.get(a, '#95a5a6') for a in agent_names])
    axes[1, 0].set_title('Directional Accuracy')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    
    # Action Distribution
    action_counts = {}
    for agent in agent_names:
        actions = action_log[agent]
        action_counts[agent] = {
            'buy': actions.count('buy'),
            'sell': actions.count('sell'),
            'hold': actions.count('hold'),
        }
    
    action_df = pd.DataFrame(action_counts).T
    action_df.plot(kind='bar', stacked=True, ax=axes[1, 1],
                  color=['#2ecc71', '#e74c3c', '#95a5a6'])
    axes[1, 1].set_title('Action Distribution')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title='Action', loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_cluster_analysis(embeddings: np.ndarray, 
                         clusters: np.ndarray,
                         method: str = 'tsne',
                         figsize: Tuple[int, int] = FIGSIZE_SQUARE) -> plt.Figure:
    """
    Plot cluster visualization using dimensionality reduction.
    
    Parameters
    ----------
    embeddings : np.ndarray
        News embeddings
    clusters : np.ndarray
        Cluster assignments
    method : str
        Dimensionality reduction method: 'tsne' or 'umap'
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)
        except ImportError:
            print("UMAP not available, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                        c=clusters, cmap='viridis', alpha=0.6, s=50)
    ax.set_title(f'News Clustering Visualization ({method.upper()})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    
    return fig


def export_figure(fig: plt.Figure, 
                 filename: str,
                 plots_dir: str = "plots",
                 dpi: int = 300) -> None:
    """
    Export figure to file.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure to export
    filename : str
        Output filename
    plots_dir : str
        Directory for plots
    dpi : int
        DPI for output
    """
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    
    output_path = plots_path / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved: {output_path}")


def display_metrics_table(metrics_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Display metrics in a formatted table.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics
    
    Returns
    -------
    df : pd.DataFrame
        Formatted metrics table
    """
    df = pd.DataFrame([
        {'Metric': k, 'Value': v}
        for k, v in metrics_dict.items()
    ])
    return df


def create_summary_table(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create summary table of experiment results.
    
    Parameters
    ----------
    results : dict
        Experiment results
    
    Returns
    -------
    summary : pd.DataFrame
        Summary table
    """
    config = results['config']
    metrics_dict = results['metrics']
    
    summary_data = {
        'Experiment': [config['experiment']['name']],
        'Dataset': [config['dataset']['type']],
        'Embedder': [config['embedder']['backend']],
        'Clusters': [config['clustering']['k']],
        'Agents': [', '.join(config['agents']['enabled'])],
        'Dir. Accuracy': [f"{metrics_dict.get('directional_accuracy', 0):.2%}"],
        'Vol. Clustering': [f"{metrics_dict.get('volatility_clustering', 0):.3f}"],
    }
    
    if 'sharpe_ratio' in metrics_dict:
        summary_data['Sharpe Ratio'] = [f"{metrics_dict['sharpe_ratio']:.2f}"]
    
    if 'max_drawdown' in metrics_dict:
        summary_data['Max Drawdown'] = [f"{metrics_dict['max_drawdown']:.2%}"]
    
    return pd.DataFrame(summary_data)


# Initialize plotting style on import
setup_plotting_style()

# Print import message
print("✓ Notebook utilities loaded successfully")
print("  - Plotting style configured")
print("  - Helper functions available")
print("  - Ready for analysis!\n")

