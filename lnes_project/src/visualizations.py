"""Visualization functions for financial metrics and analysis.

This module provides comprehensive plotting functions for:
- Equity curves with drawdown shading
- Returns distribution histograms
- Rolling metric plots
- Monthly returns heatmaps
- Underwater plots
- QQ plots for normality
- Embedding space visualizations (t-SNE, UMAP)
- Agent analysis heatmaps and networks
- Time series decomposition
- Comparative dashboards
- Correlation matrices
"""

from __future__ import annotations

from typing import Optional, Sequence, Dict, Any, Union
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from . import metrics

# Try importing optional dependencies
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Set default style
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_equity_curve(
    prices: Sequence[float],
    title: str = "Equity Curve",
    reference_prices: Optional[Sequence[float]] = None,
    show_drawdown: bool = True,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot equity curve with optional drawdown shading.
    
    Args:
        prices: Price series
        title: Plot title
        reference_prices: Optional reference prices for comparison
        show_drawdown: Whether to shade drawdown periods
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    data = metrics.prepare_equity_curve_data(prices)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot equity curve
    ax.plot(data["indices"], data["prices"], label="Portfolio", linewidth=2)
    
    # Plot reference if provided
    if reference_prices is not None:
        ax.plot(data["indices"], reference_prices, 
                label="Reference", linewidth=1.5, alpha=0.7, linestyle="--")
    
    # Shade drawdown periods
    if show_drawdown:
        ax.fill_between(
            data["indices"],
            data["cummax"],
            data["prices"],
            alpha=0.3,
            color="red",
            label="Drawdown",
        )
        ax.plot(data["indices"], data["cummax"], 
                color="green", alpha=0.5, linestyle=":", label="Peak")
    
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_underwater(
    prices: Sequence[float],
    title: str = "Underwater Plot",
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Plot underwater chart showing drawdown over time.
    
    Args:
        prices: Price series
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    data = metrics.prepare_equity_curve_data(prices)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot drawdown as area
    ax.fill_between(
        data["indices"],
        0,
        data["drawdowns"] * 100,  # Convert to percentage
        color="red",
        alpha=0.6,
    )
    ax.plot(data["indices"], data["drawdowns"] * 100, color="darkred", linewidth=1)
    
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    
    plt.tight_layout()
    return fig


def plot_returns_distribution(
    prices: Sequence[float],
    title: str = "Returns Distribution",
    bins: int = 50,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot returns distribution with normal overlay.
    
    Args:
        prices: Price series
        title: Plot title
        bins: Number of histogram bins
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    data = metrics.prepare_returns_distribution_data(prices, bins)
    
    if len(data["returns"]) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(data["returns"], bins=bins, density=True, alpha=0.7, 
            color="skyblue", edgecolor="black", label="Actual Returns")
    
    # Overlay normal distribution
    ax.plot(data["x_normal"], data["y_normal"], 
            color="red", linewidth=2, label="Normal Distribution")
    
    # Add vertical lines for mean and std
    ax.axvline(data["mu"], color="green", linestyle="--", 
               linewidth=1.5, label=f"Mean: {data['mu']:.4f}")
    ax.axvline(data["mu"] + data["sigma"], color="orange", linestyle=":", 
               linewidth=1.5, alpha=0.7, label=f"±1 Std: {data['sigma']:.4f}")
    ax.axvline(data["mu"] - data["sigma"], color="orange", linestyle=":", 
               linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_qq(
    prices: Sequence[float],
    title: str = "Q-Q Plot",
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """Plot Q-Q plot to assess normality of returns.
    
    Args:
        prices: Price series
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    returns = metrics.compute_returns(prices)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(returns) == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig
    
    # Generate Q-Q plot
    stats.probplot(returns, dist="norm", plot=ax)
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_rolling_sharpe(
    prices: Sequence[float],
    window: int = 20,
    risk_free_rate: float = 0.02,
    title: str = "Rolling Sharpe Ratio",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot rolling Sharpe ratio.
    
    Args:
        prices: Price series
        window: Rolling window size
        risk_free_rate: Risk-free rate
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    data = metrics.prepare_rolling_metric_data(
        prices, window, "sharpe", risk_free_rate
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(data["values"]) == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig
    
    ax.plot(data["indices"], data["values"], linewidth=2, color="blue")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=1, color="green", linestyle=":", linewidth=1, alpha=0.5, 
               label="Sharpe = 1")
    ax.axhline(y=2, color="darkgreen", linestyle=":", linewidth=1, alpha=0.5, 
               label="Sharpe = 2")
    
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(f"{title} (Window: {window})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_rolling_volatility(
    prices: Sequence[float],
    window: int = 20,
    title: str = "Rolling Volatility",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot rolling annualized volatility.
    
    Args:
        prices: Price series
        window: Rolling window size
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    data = metrics.prepare_rolling_metric_data(prices, window, "volatility")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(data["values"]) == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig
    
    ax.plot(data["indices"], data["values"] * 100, linewidth=2, color="red")
    
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_title(f"{title} (Window: {window})")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_monthly_returns_heatmap(
    prices: Sequence[float],
    dates: Optional[Sequence] = None,
    title: str = "Monthly Returns Heatmap",
    figsize: tuple = (12, 6),
    cmap: str = "RdYlGn",
) -> plt.Figure:
    """Plot monthly returns as a heatmap.
    
    Args:
        prices: Price series
        dates: Optional date sequence
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        
    Returns:
        Matplotlib figure
    """
    df = metrics.prepare_monthly_returns_data(prices, dates)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to percentage
    df_pct = df * 100
    
    # Create heatmap
    im = ax.imshow(df_pct.values, cmap=cmap, aspect="auto", vmin=-10, vmax=10)
    
    # Set ticks
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Return (%)", rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df_pct.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 5 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", 
                       color=text_color, fontsize=8)
    
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_agent_comparison(
    action_log: metrics.ActionLog,
    price_series: Sequence[float],
    metric: str = "pnl",
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot comparison of agent performance.
    
    Args:
        action_log: Agent action log
        price_series: Price series
        metric: Metric to compare ("pnl", "win_rate", "profit_factor")
        title: Plot title (auto-generated if None)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if metric == "pnl":
        df = metrics.agent_profitability(action_log, price_series)
        y_col = "pnl"
        y_label = "Profit & Loss"
        if title is None:
            title = "Agent Profitability Comparison"
    elif metric == "win_rate":
        df = metrics.win_rate(action_log, price_series)
        y_col = "win_rate"
        y_label = "Win Rate"
        if title is None:
            title = "Agent Win Rate Comparison"
    elif metric == "profit_factor":
        df = metrics.profit_factor(action_log, price_series)
        y_col = "profit_factor"
        y_label = "Profit Factor"
        if title is None:
            title = "Agent Profit Factor Comparison"
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Sort by metric
    df = df.sort_values(y_col, ascending=False)
    
    # Create bar chart
    colors = ["green" if v > 0 else "red" for v in df[y_col]]
    ax.bar(df["agent"], df[y_col], color=colors, alpha=0.7, edgecolor="black")
    
    ax.set_xlabel("Agent")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Rotate x labels if needed
    if len(df) > 5:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add horizontal line at zero
    if metric in ["pnl", "profit_factor"]:
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    return fig


def plot_comprehensive_dashboard(
    prices: Sequence[float],
    action_log: Optional[metrics.ActionLog] = None,
    reference_prices: Optional[Sequence[float]] = None,
    title: str = "Performance Dashboard",
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """Create comprehensive dashboard with multiple plots.
    
    Args:
        prices: Price series
        action_log: Optional agent action log
        reference_prices: Optional reference prices
        title: Dashboard title
        figsize: Figure size
        
    Returns:
        Matplotlib figure with subplots
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Equity curve with drawdown
    ax1 = fig.add_subplot(gs[0, :])
    data = metrics.prepare_equity_curve_data(prices)
    ax1.plot(data["indices"], data["prices"], label="Portfolio", linewidth=2)
    if reference_prices is not None:
        ax1.plot(data["indices"], reference_prices, 
                label="Reference", linewidth=1.5, alpha=0.7, linestyle="--")
    ax1.fill_between(data["indices"], data["cummax"], data["prices"], 
                     alpha=0.3, color="red")
    ax1.set_ylabel("Price")
    ax1.set_title("Equity Curve with Drawdown")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    
    # 2. Underwater plot
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(data["indices"], 0, data["drawdowns"] * 100, 
                     color="red", alpha=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Underwater Plot")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    
    # 3. Returns distribution
    ax3 = fig.add_subplot(gs[2, 0])
    dist_data = metrics.prepare_returns_distribution_data(prices, bins=30)
    if len(dist_data["returns"]) > 0:
        ax3.hist(dist_data["returns"], bins=30, density=True, 
                alpha=0.7, color="skyblue", edgecolor="black")
        ax3.plot(dist_data["x_normal"], dist_data["y_normal"], 
                color="red", linewidth=2)
    ax3.set_xlabel("Return")
    ax3.set_ylabel("Density")
    ax3.set_title("Returns Distribution")
    ax3.grid(True, alpha=0.3)
    
    # 4. Agent comparison (if available)
    ax4 = fig.add_subplot(gs[2, 1])
    if action_log is not None:
        prof_df = metrics.agent_profitability(action_log, prices)
        prof_df = prof_df.sort_values("pnl", ascending=False)
        colors = ["green" if v > 0 else "red" for v in prof_df["pnl"]]
        ax4.bar(prof_df["agent"], prof_df["pnl"], color=colors, alpha=0.7)
        ax4.set_ylabel("PnL")
        ax4.set_title("Agent Profitability")
        ax4.grid(True, alpha=0.3, axis="y")
        if len(prof_df) > 3:
            plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
    else:
        ax4.text(0.5, 0.5, "No agent data", ha="center", va="center")
        ax4.set_title("Agent Profitability")
    
    return fig


# ============================================================================
# EMBEDDING VISUALIZATIONS
# ============================================================================

def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    texts: Optional[Sequence[str]] = None,
    title: str = "t-SNE Embedding Visualization",
    perplexity: int = 30,
    random_state: int = 42,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Visualize embeddings using t-SNE dimensionality reduction.
    
    Args:
        embeddings: High-dimensional embeddings (N x D)
        labels: Cluster labels for coloring
        texts: Text labels for annotation (optional)
        title: Plot title
        perplexity: t-SNE perplexity parameter
        random_state: Random seed
        figsize: Figure size
        
    Returns:
        Matplotlib figure
        
    References:
        van der Maaten & Hinton (2008). "Visualizing Data using t-SNE"
    """
    if not TSNE_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "t-SNE not available. Install scikit-learn.", 
                ha="center", va="center")
        return fig
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1),
                random_state=random_state, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        # Color by cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[i]], label=f"Cluster {label}", 
                      alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        
        ax.legend(loc="best", framealpha=0.9)
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                  alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # Add text annotations if provided (for first 10 points)
    if texts is not None:
        for i in range(min(10, len(texts))):
            ax.annotate(texts[i][:30] + "...", 
                       (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       fontsize=7, alpha=0.7)
    
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_embedding_umap(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    texts: Optional[Sequence[str]] = None,
    title: str = "UMAP Embedding Visualization",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Visualize embeddings using UMAP dimensionality reduction.
    
    Args:
        embeddings: High-dimensional embeddings (N x D)
        labels: Cluster labels for coloring
        texts: Text labels for annotation (optional)
        title: Plot title
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed
        figsize: Figure size
        
    Returns:
        Matplotlib figure
        
    References:
        McInnes et al. (2018). "UMAP: Uniform Manifold Approximation and Projection"
    """
    if not UMAP_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "UMAP not available. Install umap-learn.", 
                ha="center", va="center")
        return fig
    
    # Perform UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                        min_dist=min_dist, random_state=random_state)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        # Color by cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[i]], label=f"Cluster {label}", 
                      alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        
        ax.legend(loc="best", framealpha=0.9)
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                  alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # Add text annotations if provided (for first 10 points)
    if texts is not None:
        for i in range(min(10, len(texts))):
            ax.annotate(texts[i][:30] + "...", 
                       (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       fontsize=7, alpha=0.7)
    
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_cluster_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "Silhouette Analysis",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot silhouette analysis for clustering quality.
    
    Args:
        embeddings: High-dimensional embeddings
        labels: Cluster labels
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
        
    References:
        Rousseeuw (1987). "Silhouettes: A graphical aid to the interpretation
        and validation of cluster analysis"
    """
    try:
        from sklearn.metrics import silhouette_samples, silhouette_score
    except ImportError:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "scikit-learn not available", ha="center", va="center")
        return fig
    
    # Compute silhouette scores
    silhouette_avg = silhouette_score(embeddings, labels)
    sample_silhouette_values = silhouette_samples(embeddings, labels)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_lower = 10
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        # Aggregate silhouette scores for samples in this cluster
        cluster_silhouette_values = sample_silhouette_values[labels == label]
        cluster_silhouette_values.sort()
        
        size_cluster = len(cluster_silhouette_values)
        y_upper = y_lower + size_cluster
        
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        
        # Label the silhouette plots with their cluster numbers
        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(label))
        
        y_lower = y_upper + 10
    
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")
    ax.set_title(f"{title}\n(Avg Score: {silhouette_avg:.3f})")
    
    # Add vertical line for average score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
               label=f"Avg: {silhouette_avg:.3f}")
    
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    return fig


# ============================================================================
# AGENT ANALYSIS VISUALIZATIONS
# ============================================================================

def plot_agent_action_heatmap(
    action_log: Dict[str, list],
    dates: Optional[Sequence] = None,
    title: str = "Agent Action Heatmap",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Plot heatmap of agent actions over time.
    
    Args:
        action_log: Dictionary mapping agent names to action sequences
        dates: Optional date sequence for x-axis
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numeric: buy=1, hold=0, sell=-1
    action_to_num = {"buy": 1, "hold": 0, "sell": -1}
    
    agents = sorted(action_log.keys())
    n_periods = len(action_log[agents[0]])
    
    # Create matrix
    matrix = np.zeros((len(agents), n_periods))
    for i, agent in enumerate(agents):
        for j, action in enumerate(action_log[agent]):
            matrix[i, j] = action_to_num.get(action, 0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    
    # Set ticks
    ax.set_yticks(np.arange(len(agents)))
    ax.set_yticklabels(agents)
    
    if dates is not None:
        # Show every nth date to avoid crowding
        step = max(1, n_periods // 10)
        ax.set_xticks(np.arange(0, n_periods, step))
        ax.set_xticklabels([str(dates[i])[:10] if i < len(dates) else "" 
                           for i in range(0, n_periods, step)], rotation=45, ha="right")
    else:
        ax.set_xlabel("Time Period")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["Sell", "Hold", "Buy"])
    
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_agent_correlation_network(
    action_log: Dict[str, list],
    title: str = "Agent Decision Correlation Network",
    threshold: float = 0.3,
    figsize: tuple = (10, 10),
) -> plt.Figure:
    """Plot network graph of agent decision correlations.
    
    Args:
        action_log: Dictionary mapping agent names to action sequences
        title: Plot title
        threshold: Minimum correlation to show edge
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not NETWORKX_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "NetworkX not available. Install networkx.", 
                ha="center", va="center")
        return fig
    
    # Convert actions to numeric
    action_to_num = {"buy": 1, "hold": 0, "sell": -1}
    agents = sorted(action_log.keys())
    
    # Create matrix
    matrix = np.zeros((len(agents), len(action_log[agents[0]])))
    for i, agent in enumerate(agents):
        for j, action in enumerate(action_log[agent]):
            matrix[i, j] = action_to_num.get(action, 0)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(matrix)
    
    # Create network
    G = nx.Graph()
    for i, agent in enumerate(agents):
        G.add_node(agent)
    
    # Add edges for correlations above threshold
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if abs(corr_matrix[i, j]) > threshold:
                G.add_edge(agents[i], agents[j], weight=corr_matrix[i, j])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.9, ax=ax)
    
    # Draw edges with width proportional to correlation
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Separate positive and negative correlations
    pos_edges = [(u, v) for (u, v), w in zip(edges, weights) if w > 0]
    neg_edges = [(u, v) for (u, v), w in zip(edges, weights) if w < 0]
    pos_weights = [abs(w) * 3 for w in weights if w > 0]
    neg_weights = [abs(w) * 3 for w in weights if w < 0]
    
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=pos_weights,
                          edge_color='green', alpha=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=neg_weights,
                          edge_color='red', alpha=0.6, ax=ax, style='dashed')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title(title)
    ax.axis('off')
    
    # Add legend
    green_patch = mpatches.Patch(color='green', label='Positive Correlation')
    red_patch = mpatches.Patch(color='red', label='Negative Correlation')
    ax.legend(handles=[green_patch, red_patch], loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_action_frequency(
    action_log: Dict[str, list],
    title: str = "Action Frequency by Agent",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot action frequency distribution for each agent.
    
    Args:
        action_log: Dictionary mapping agent names to action sequences
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    agents = sorted(action_log.keys())
    actions = ["buy", "hold", "sell"]
    
    # Count actions for each agent
    data = []
    for agent in agents:
        counts = {action: action_log[agent].count(action) for action in actions}
        data.append(counts)
    
    df = pd.DataFrame(data, index=agents)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Stacked bar chart
    df.plot(kind='bar', stacked=True, ax=ax, 
            color=['green', 'gray', 'red'], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel("Agent")
    ax.set_ylabel("Number of Actions")
    ax.set_title(title)
    ax.legend(title="Action", loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_agent_rolling_pnl(
    action_log: Dict[str, list],
    price_series: Sequence[float],
    window: int = 10,
    title: str = "Rolling P&L by Agent",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot rolling P&L for each agent.
    
    Args:
        action_log: Dictionary mapping agent names to action sequences
        price_series: Price series
        window: Rolling window size
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    agents = sorted(action_log.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))
    
    for i, agent in enumerate(agents):
        # Compute cumulative PnL
        pnl = []
        position = 0
        cumulative_pnl = 0
        
        for j, action in enumerate(action_log[agent]):
            if j == 0:
                pnl.append(0)
                continue
            
            # Price change
            price_change = price_series[j] - price_series[j - 1]
            
            # Update PnL based on previous position
            cumulative_pnl += position * price_change
            
            # Update position
            if action == "buy":
                position += 1
            elif action == "sell":
                position -= 1
            
            pnl.append(cumulative_pnl)
        
        # Compute rolling sum
        pnl_series = pd.Series(pnl)
        rolling_pnl = pnl_series.rolling(window=window, min_periods=1).sum()
        
        ax.plot(rolling_pnl, label=agent, linewidth=2, color=colors[i], alpha=0.8)
    
    ax.set_xlabel("Time Period")
    ax.set_ylabel(f"Rolling P&L (Window: {window})")
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    return fig


# ============================================================================
# TIME SERIES VISUALIZATIONS
# ============================================================================

def plot_order_flow_waterfall(
    order_flow: Sequence[float],
    dates: Optional[Sequence] = None,
    title: str = "Order Flow Waterfall",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Plot order flow as a waterfall chart.
    
    Args:
        order_flow: Sequence of net order flow values
        dates: Optional date sequence
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n = len(order_flow)
    x = np.arange(n)
    
    # Color bars based on positive/negative
    colors = ['green' if val > 0 else 'red' if val < 0 else 'gray' 
              for val in order_flow]
    
    ax.bar(x, order_flow, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    if dates is not None:
        step = max(1, n // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(dates[i])[:10] for i in range(0, n, step)], 
                          rotation=45, ha="right")
    else:
        ax.set_xlabel("Time Period")
    
    ax.set_ylabel("Net Order Flow")
    ax.set_title(title)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig


def plot_regime_detection(
    prices: Sequence[float],
    regime_labels: Optional[np.ndarray] = None,
    title: str = "Price Regime Detection",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Plot prices with regime change detection.
    
    Args:
        prices: Price series
        regime_labels: Regime labels for each period
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(prices))
    
    if regime_labels is not None:
        # Color-code by regime
        unique_regimes = np.unique(regime_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
        
        for i, regime in enumerate(unique_regimes):
            mask = regime_labels == regime
            ax.plot(x[mask], np.array(prices)[mask], 'o', color=colors[i],
                   label=f"Regime {regime}", markersize=6, alpha=0.7)
        
        # Draw connecting lines
        ax.plot(x, prices, color='black', alpha=0.3, linewidth=1, zorder=0)
        
        ax.legend(loc="best")
    else:
        ax.plot(x, prices, linewidth=2)
    
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# COMPARATIVE VISUALIZATIONS
# ============================================================================

def plot_metric_comparison_radar(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[list] = None,
    title: str = "Agent Metrics Comparison (Radar)",
    figsize: tuple = (10, 10),
) -> plt.Figure:
    """Plot radar chart comparing multiple metrics across agents.
    
    Args:
        metrics_dict: Nested dict {agent: {metric: value}}
        metrics_to_plot: List of metric names to include
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if metrics_to_plot is None:
        # Use all metrics from first agent
        first_agent = list(metrics_dict.keys())[0]
        metrics_to_plot = list(metrics_dict[first_agent].keys())
    
    agents = list(metrics_dict.keys())
    num_vars = len(metrics_to_plot)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))
    
    for i, agent in enumerate(agents):
        values = [metrics_dict[agent].get(m, 0) for m in metrics_to_plot]
        values += values[:1]  # Close the plot
        
        ax.plot(angles, values, 'o-', linewidth=2, label=agent, 
                color=colors[i], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    ax.set_title(title, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    data: Union[pd.DataFrame, Dict[str, list]],
    title: str = "Correlation Matrix",
    figsize: tuple = (10, 8),
    annot: bool = True,
) -> plt.Figure:
    """Plot correlation matrix heatmap.
    
    Args:
        data: DataFrame or dict of sequences
        title: Plot title
        figsize: Figure size
        annot: Whether to annotate cells with values
        
    Returns:
        Matplotlib figure
    """
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data
    
    # Compute correlation
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(corr, annot=annot, cmap='RdYlGn', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)
    
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_figure(
    fig: plt.Figure,
    filepath: str,
    dpi: int = 300,
    formats: list = ['png'],
    close: bool = False,
) -> None:
    """Save figure to file(s) in multiple formats.
    
    Args:
        fig: Matplotlib figure
        filepath: Base filepath (without extension)
        dpi: Resolution for raster formats
        formats: List of formats ('png', 'pdf', 'svg')
        close: Whether to close figure after saving
    """
    import os
    
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    for fmt in formats:
        output_path = f"{filepath}.{fmt}"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   format=fmt, transparent=False)
        print(f"Saved: {output_path}")
    
    if close:
        plt.close(fig)


def set_publication_style(font_family: str = 'serif', font_size: int = 10) -> None:
    """Set matplotlib style for publication-quality figures.
    
    Args:
        font_family: Font family ('serif', 'sans-serif')
        font_size: Base font size
    """
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = font_size + 1
    plt.rcParams['axes.titlesize'] = font_size + 2
    plt.rcParams['xtick.labelsize'] = font_size - 1
    plt.rcParams['ytick.labelsize'] = font_size - 1
    plt.rcParams['legend.fontsize'] = font_size - 1
    plt.rcParams['figure.titlesize'] = font_size + 3
    
    # Use LaTeX if available
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    except:
        warnings.warn("LaTeX not available for rendering")


def reset_style() -> None:
    """Reset matplotlib to default style."""
    plt.rcdefaults()
    sns.set_style("whitegrid")
    sns.set_palette("husl")

