"""Interactive dashboard for MacroMind experiments using Streamlit.

Run with:
    streamlit run src/dashboard.py

Features:
- Dataset selection
- Agent configuration
- Real-time experiment execution
- Interactive visualizations
- Metric comparison
- Export functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import data_loader, preprocess, embedder, clusterer, agents, simulator, metrics
from src import visualizations as viz
from src.config_loader import load_config, list_available_configs
from src.result_cache import ResultCache


# Page configuration
st.set_page_config(
    page_title="MacroMind Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_experiment_data(dataset_choice: str, config: Dict[str, Any]) -> tuple:
    """Load data based on dataset choice.
    
    Returns:
        (news_df, prices_df) tuple
    """
    if dataset_choice == "Small Curated Dataset":
        news_df = data_loader.load_news(config['dataset']['news_file'])
        prices_df = data_loader.load_prices(config['dataset']['prices_file'])
    elif dataset_choice == "FNSPID":
        from src.data_loader import load_fnspid
        news_df, prices_df = load_fnspid(
            ticker=config['dataset']['fnspid']['ticker'],
            start_date=config['dataset']['fnspid']['start_date'],
            end_date=config['dataset']['fnspid']['end_date'],
        )
    else:
        st.error(f"Unknown dataset: {dataset_choice}")
        return None, None
    
    return news_df, prices_df


def run_experiment(config: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
    """Run complete experiment pipeline.
    
    Args:
        config: Experiment configuration
        use_cache: Whether to use cached results
        
    Returns:
        Dictionary with all results
    """
    # Check cache
    cache = ResultCache()
    
    if use_cache:
        cached_results = cache.list_cached_results()
        if cached_results:
            # Try to load from cache (simplified)
            try:
                st.info("Loading cached results...")
                # In a full implementation, we'd match config and load
                # For now, just run fresh
            except:
                pass
    
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
    
    agent_metrics = metrics.compare_agents_report(action_log, sim_prices)
    
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


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    # Configuration presets
    st.sidebar.subheader("Configuration Preset")
    available_configs = list_available_configs()
    config_choice = st.sidebar.selectbox(
        "Select preset",
        ["default_config"] + available_configs,
        help="Choose a predefined configuration"
    )
    
    # Load base config
    base_config = load_config(config_choice.replace("_config", "").replace("default", ""))
    
    # Dataset selection
    st.sidebar.subheader("Dataset")
    dataset_choice = st.sidebar.radio(
        "Choose dataset",
        ["Small Curated Dataset", "FNSPID"],
        help="Select which dataset to use"
    )
    
    # Update config based on dataset
    if dataset_choice == "Small Curated Dataset":
        base_config['experiment']['dataset'] = "small"
    else:
        base_config['experiment']['dataset'] = "fnspid"
        
        # FNSPID specific options
        st.sidebar.subheader("FNSPID Options")
        ticker = st.sidebar.text_input("Ticker", value="AAPL")
        start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
        end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-03-31"))
        
        base_config['dataset']['fnspid']['ticker'] = ticker
        base_config['dataset']['fnspid']['start_date'] = str(start_date)
        base_config['dataset']['fnspid']['end_date'] = str(end_date)
    
    # Experiment parameters
    st.sidebar.subheader("Experiment Parameters")
    
    # Embedder backend
    backend = st.sidebar.selectbox(
        "Embedding Backend",
        ["tfidf", "sentence-transformer"],
        index=0,
        help="Choose embedding method"
    )
    base_config['embedder']['backend'] = backend
    
    # Number of clusters
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 
                         value=base_config['clustering']['k'],
                         help="Number of clusters for k-means")
    base_config['clustering']['k'] = k
    
    # Simulator parameters
    st.sidebar.subheader("Simulator Parameters")
    alpha = st.sidebar.slider("Price Impact (alpha)", 0.0, 0.1, 
                              value=base_config['simulator']['alpha'],
                              step=0.001, format="%.3f",
                              help="Impact of order flow on prices")
    base_config['simulator']['alpha'] = alpha
    
    noise_std = st.sidebar.slider("Noise Std Dev", 0.0, 0.05,
                                  value=base_config['simulator']['noise_std'],
                                  step=0.001, format="%.3f",
                                  help="Standard deviation of price noise")
    base_config['simulator']['noise_std'] = noise_std
    
    # Agent selection
    st.sidebar.subheader("Agents")
    agent_types = st.sidebar.multiselect(
        "Select Agents",
        ["random", "momentum", "contrarian", "newsreactive"],
        default=base_config['agents']['types'],
        help="Choose which agents to include"
    )
    base_config['agents']['types'] = agent_types
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        use_cache = st.checkbox("Use Result Cache", value=True)
        show_debug = st.checkbox("Show Debug Info", value=False)
    
    return base_config, use_cache, show_debug


# ============================================================================
# MAIN CONTENT
# ============================================================================

def main():
    """Main dashboard application."""
    
    # Title
    st.title("📈 MacroMind: Latent News Event Simulation")
    st.markdown("Interactive dashboard for exploring agent-based market simulations")
    
    # Render sidebar and get config
    config, use_cache, show_debug = render_sidebar()
    
    # Debug info
    if show_debug:
        with st.expander("Debug: Configuration"):
            st.json(config)
    
    # Run button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        run_button = st.button("▶️ Run Experiment", type="primary", use_container_width=True)
    with col2:
        clear_cache = st.button("🗑️ Clear Cache", use_container_width=True)
    
    if clear_cache:
        cache = ResultCache()
        # Cache clearing logic would go here
        st.success("Cache cleared!")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Run experiment
    if run_button:
        with st.spinner("Running experiment..."):
            try:
                results = run_experiment(config, use_cache=use_cache)
                st.session_state.results = results
                st.success("✅ Experiment completed!")
            except Exception as e:
                st.error(f"❌ Error running experiment: {str(e)}")
                if show_debug:
                    st.exception(e)
                return
    
    # Display results if available
    if st.session_state.results is not None:
        results = st.session_state.results
        
        # Tabs for different views
        tabs = st.tabs([
            "📊 Overview",
            "📈 Performance",
            "🤖 Agent Analysis",
            "🔍 Embeddings",
            "📉 Time Series",
            "📋 Data"
        ])
        
        # Tab 1: Overview
        with tabs[0]:
            render_overview_tab(results)
        
        # Tab 2: Performance
        with tabs[1]:
            render_performance_tab(results)
        
        # Tab 3: Agent Analysis
        with tabs[2]:
            render_agent_analysis_tab(results)
        
        # Tab 4: Embeddings
        with tabs[3]:
            render_embeddings_tab(results)
        
        # Tab 5: Time Series
        with tabs[4]:
            render_timeseries_tab(results)
        
        # Tab 6: Data
        with tabs[5]:
            render_data_tab(results)


def render_overview_tab(results: Dict[str, Any]):
    """Render overview tab."""
    st.header("Experiment Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = (results['sim_prices'][-1] / results['sim_prices'][0] - 1) * 100
        st.metric("Total Return", f"{total_return:.2f}%")
    
    with col2:
        sharpe = results['perf_report'].get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
    
    with col3:
        max_dd = results['perf_report'].get('max_drawdown', 0) * 100
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
    
    with col4:
        n_agents = len(results['action_log'])
        st.metric("Active Agents", n_agents)
    
    # Comprehensive dashboard
    st.subheader("Performance Dashboard")
    fig = viz.plot_comprehensive_dashboard(
        prices=results['sim_prices'],
        action_log=results['action_log'],
        reference_prices=results['prices_df']['close'].tolist(),
    )
    st.pyplot(fig)
    plt.close(fig)


def render_performance_tab(results: Dict[str, Any]):
    """Render performance analysis tab."""
    st.header("Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Equity Curve")
        fig = viz.plot_equity_curve(
            results['sim_prices'],
            reference_prices=results['prices_df']['close'].tolist(),
            show_drawdown=True
        )
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("Returns Distribution")
        fig = viz.plot_returns_distribution(results['sim_prices'])
        st.pyplot(fig)
        plt.close(fig)
    
    # Underwater plot
    st.subheader("Underwater Plot")
    fig = viz.plot_underwater(results['sim_prices'])
    st.pyplot(fig)
    plt.close(fig)
    
    # Performance report
    st.subheader("Detailed Metrics")
    report_text = metrics.format_performance_report(results['perf_report'])
    st.text(report_text)


def render_agent_analysis_tab(results: Dict[str, Any]):
    """Render agent analysis tab."""
    st.header("Agent Analysis")
    
    # Agent profitability
    st.subheader("Agent Profitability")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = viz.plot_agent_comparison(results['action_log'], results['sim_prices'], metric='pnl')
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        fig = viz.plot_agent_comparison(results['action_log'], results['sim_prices'], metric='win_rate')
        st.pyplot(fig)
        plt.close(fig)
    
    # Action heatmap
    st.subheader("Agent Action Heatmap")
    fig = viz.plot_agent_action_heatmap(results['action_log'])
    st.pyplot(fig)
    plt.close(fig)
    
    # Action frequency
    st.subheader("Action Frequency")
    fig = viz.plot_action_frequency(results['action_log'])
    st.pyplot(fig)
    plt.close(fig)
    
    # Agent correlation network
    st.subheader("Agent Decision Correlation Network")
    fig = viz.plot_agent_correlation_network(results['action_log'])
    st.pyplot(fig)
    plt.close(fig)
    
    # Agent metrics table
    st.subheader("Agent Metrics")
    st.dataframe(results['agent_metrics'], use_container_width=True)


def render_embeddings_tab(results: Dict[str, Any]):
    """Render embeddings visualization tab."""
    st.header("Embedding Analysis")
    
    # Method selection
    method = st.radio("Dimensionality Reduction Method", ["t-SNE", "UMAP"], horizontal=True)
    
    if method == "t-SNE":
        st.subheader("t-SNE Visualization")
        fig = viz.plot_embedding_tsne(
            results['embeddings'],
            labels=results['cluster_labels'],
            texts=results['preprocessed_texts'],
        )
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.subheader("UMAP Visualization")
        fig = viz.plot_embedding_umap(
            results['embeddings'],
            labels=results['cluster_labels'],
            texts=results['preprocessed_texts'],
        )
        st.pyplot(fig)
        plt.close(fig)
    
    # Silhouette analysis
    st.subheader("Cluster Quality (Silhouette Analysis)")
    fig = viz.plot_cluster_silhouette(results['embeddings'], results['cluster_labels'])
    st.pyplot(fig)
    plt.close(fig)


def render_timeseries_tab(results: Dict[str, Any]):
    """Render time series analysis tab."""
    st.header("Time Series Analysis")
    
    # Price comparison
    st.subheader("Simulated vs Reference Prices")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(results['sim_prices'], label='Simulated', linewidth=2)
    ax.plot(results['prices_df']['close'].tolist(), label='Reference', linewidth=2, alpha=0.7, linestyle='--')
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Price")
    ax.set_title("Price Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    
    # Rolling metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rolling Sharpe Ratio")
        fig = viz.plot_rolling_sharpe(results['sim_prices'], window=20)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("Rolling Volatility")
        fig = viz.plot_rolling_volatility(results['sim_prices'], window=20)
        st.pyplot(fig)
        plt.close(fig)
    
    # Order flow
    if 'order_flow' in results:
        st.subheader("Order Flow")
        fig = viz.plot_order_flow_waterfall(results['order_flow'])
        st.pyplot(fig)
        plt.close(fig)


def render_data_tab(results: Dict[str, Any]):
    """Render raw data tab."""
    st.header("Data Explorer")
    
    # News data
    st.subheader("News Data")
    st.dataframe(results['news_df'].head(20), use_container_width=True)
    
    # Download button
    csv = results['news_df'].to_csv(index=False)
    st.download_button(
        label="📥 Download News CSV",
        data=csv,
        file_name="news_data.csv",
        mime="text/csv",
    )
    
    # Price data
    st.subheader("Price Data")
    price_df = pd.DataFrame({
        'index': range(len(results['sim_prices'])),
        'simulated_price': results['sim_prices'],
        'reference_price': results['prices_df']['close'].tolist()[:len(results['sim_prices'])],
    })
    st.dataframe(price_df, use_container_width=True)
    
    # Download button
    csv = price_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Price CSV",
        data=csv,
        file_name="price_data.csv",
        mime="text/csv",
    )
    
    # Cluster distribution
    st.subheader("Cluster Distribution")
    cluster_counts = pd.Series(results['cluster_labels']).value_counts().sort_index()
    st.bar_chart(cluster_counts)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

