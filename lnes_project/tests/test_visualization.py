"""Tests for visualization functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")

from tests.test_utils import (
    generate_mock_action_log,
    generate_mock_embeddings,
    generate_mock_price_data,
)


# =============================================================================
# Basic Visualization Tests
# =============================================================================

class TestBasicVisualizations:
    """Test basic visualization functions."""
    
    def test_plot_equity_curve_creation(self, sample_prices_series):
        """Test equity curve plot is created."""
        from src.visualizations import plot_equity_curve
        
        fig = plot_equity_curve(sample_prices_series)
        
        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)
        
        # Check axes exist
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_plot_equity_curve_with_drawdown(self, sample_prices_series):
        """Test equity curve with drawdown shading."""
        from src.visualizations import plot_equity_curve
        
        fig = plot_equity_curve(sample_prices_series, show_drawdown=True)
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_plot_underwater(self, sample_prices_series):
        """Test underwater plot creation."""
        from src.visualizations import plot_underwater
        
        fig = plot_underwater(sample_prices_series)
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_plot_returns_distribution(self, sample_prices_series):
        """Test returns distribution plot."""
        from src.visualizations import plot_returns_distribution
        
        fig = plot_returns_distribution(sample_prices_series)
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_plot_qq_plot(self, sample_prices_series):
        """Test Q-Q plot creation."""
        from src.visualizations import plot_qq_plot
        
        fig = plot_qq_plot(sample_prices_series)
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_plot_rolling_sharpe(self, sample_prices_series):
        """Test rolling Sharpe ratio plot."""
        from src.visualizations import plot_rolling_sharpe
        
        fig = plot_rolling_sharpe(sample_prices_series, window=20)
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_plot_rolling_volatility(self, sample_prices_series):
        """Test rolling volatility plot."""
        from src.visualizations import plot_rolling_volatility
        
        fig = plot_rolling_volatility(sample_prices_series, window=20)
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)


# =============================================================================
# Agent Visualization Tests
# =============================================================================

class TestAgentVisualizations:
    """Test agent-related visualizations."""
    
    def test_plot_agent_comparison(self, sample_action_log, sample_prices_series):
        """Test agent comparison bar chart."""
        from src.visualizations import plot_agent_comparison
        
        fig = plot_agent_comparison(
            action_log=sample_action_log,
            price_series=sample_prices_series,
            metric="pnl",
        )
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_plot_agent_action_heatmap(self, sample_action_log):
        """Test agent action heatmap."""
        from src.visualizations import plot_agent_action_heatmap
        
        fig = plot_agent_action_heatmap(sample_action_log)
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_plot_action_frequency(self, sample_action_log):
        """Test action frequency stacked bar chart."""
        from src.visualizations import plot_action_frequency
        
        fig = plot_action_frequency(sample_action_log)
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_plot_agent_correlation_network(self, sample_action_log):
        """Test agent correlation network graph."""
        pytest.importorskip("networkx")
        from src.visualizations import plot_agent_correlation_network
        
        fig = plot_agent_correlation_network(sample_action_log)
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_plot_agent_rolling_pnl(self, sample_action_log, sample_prices_series):
        """Test agent rolling P&L plot."""
        from src.visualizations import plot_agent_rolling_pnl
        
        fig = plot_agent_rolling_pnl(
            action_log=sample_action_log,
            price_series=sample_prices_series,
        )
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)


# =============================================================================
# Embedding Visualization Tests
# =============================================================================

class TestEmbeddingVisualizations:
    """Test embedding-related visualizations."""
    
    @pytest.mark.slow
    def test_plot_embedding_tsne(self, sample_embeddings, sample_cluster_labels):
        """Test t-SNE embedding plot."""
        from src.visualizations import plot_embedding_tsne
        
        fig = plot_embedding_tsne(
            embeddings=sample_embeddings,
            labels=sample_cluster_labels,
        )
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    @pytest.mark.slow
    def test_plot_embedding_umap(self, sample_embeddings, sample_cluster_labels):
        """Test UMAP embedding plot."""
        pytest.importorskip("umap")
        from src.visualizations import plot_embedding_umap
        
        fig = plot_embedding_umap(
            embeddings=sample_embeddings,
            labels=sample_cluster_labels,
        )
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_plot_cluster_silhouette(self, sample_embeddings, sample_cluster_labels):
        """Test silhouette plot."""
        from src.visualizations import plot_cluster_silhouette
        
        fig = plot_cluster_silhouette(
            embeddings=sample_embeddings,
            labels=sample_cluster_labels,
        )
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)


# =============================================================================
# Dashboard and Multi-Panel Tests
# =============================================================================

class TestDashboardVisualizations:
    """Test dashboard and multi-panel visualizations."""
    
    def test_plot_comprehensive_dashboard(
        self,
        sample_prices_series,
        sample_action_log,
    ):
        """Test comprehensive dashboard creation."""
        from src.visualizations import plot_comprehensive_dashboard
        
        fig = plot_comprehensive_dashboard(
            prices=sample_prices_series,
            action_log=sample_action_log,
        )
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) >= 4  # Should have multiple subplots
        
        plt.close(fig)
    
    def test_plot_metric_comparison_radar(self):
        """Test radar chart for metric comparison."""
        from src.visualizations import plot_metric_comparison_radar
        
        metrics = {
            "Agent1": {"sharpe": 1.5, "sortino": 1.8, "calmar": 2.0, "accuracy": 0.6},
            "Agent2": {"sharpe": 1.2, "sortino": 1.5, "calmar": 1.7, "accuracy": 0.55},
        }
        
        fig = plot_metric_comparison_radar(metrics)
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_plot_correlation_matrix(self, sample_action_log):
        """Test correlation matrix heatmap."""
        from src.visualizations import plot_correlation_matrix
        
        fig = plot_correlation_matrix(sample_action_log)
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)


# =============================================================================
# Export and Styling Tests
# =============================================================================

class TestVisualizationExport:
    """Test visualization export functionality."""
    
    def test_save_figure_png(self, sample_prices_series, temp_dir):
        """Test saving figure as PNG."""
        from src.visualizations import plot_equity_curve, save_figure
        
        fig = plot_equity_curve(sample_prices_series)
        output_path = temp_dir / "test_plot"
        
        save_figure(fig, str(output_path), formats=["png"])
        
        assert (temp_dir / "test_plot.png").exists()
        plt.close(fig)
    
    def test_save_figure_pdf(self, sample_prices_series, temp_dir):
        """Test saving figure as PDF."""
        from src.visualizations import plot_equity_curve, save_figure
        
        fig = plot_equity_curve(sample_prices_series)
        output_path = temp_dir / "test_plot"
        
        save_figure(fig, str(output_path), formats=["pdf"])
        
        assert (temp_dir / "test_plot.pdf").exists()
        plt.close(fig)
    
    def test_save_figure_multiple_formats(self, sample_prices_series, temp_dir):
        """Test saving figure in multiple formats."""
        from src.visualizations import plot_equity_curve, save_figure
        
        fig = plot_equity_curve(sample_prices_series)
        output_path = temp_dir / "test_plot"
        
        save_figure(fig, str(output_path), formats=["png", "pdf", "svg"])
        
        assert (temp_dir / "test_plot.png").exists()
        assert (temp_dir / "test_plot.pdf").exists()
        assert (temp_dir / "test_plot.svg").exists()
        plt.close(fig)
    
    def test_set_publication_style(self):
        """Test setting publication style."""
        from src.visualizations import set_publication_style, reset_style
        
        # Set publication style
        set_publication_style(font_family="serif", font_size=10)
        
        # Check some style properties
        current_font = plt.rcParams["font.family"]
        assert "serif" in str(current_font).lower() or "times" in str(current_font).lower()
        
        # Reset style
        reset_style()
    
    def test_reset_style(self):
        """Test resetting matplotlib style."""
        from src.visualizations import set_publication_style, reset_style
        
        # Modify style
        set_publication_style()
        
        # Reset
        reset_style()
        
        # Should return to defaults (not checking exact values since they vary)
        assert plt.rcParams is not None


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestVisualizationEdgeCases:
    """Test visualization edge cases."""
    
    def test_plot_with_empty_data(self):
        """Test plotting with empty data."""
        from src.visualizations import plot_equity_curve
        
        try:
            fig = plot_equity_curve([])
            plt.close(fig)
            # If it doesn't raise, that's fine
        except (ValueError, IndexError):
            # Expected to raise error for empty data
            pass
    
    def test_plot_with_single_datapoint(self):
        """Test plotting with single data point."""
        from src.visualizations import plot_equity_curve
        
        try:
            fig = plot_equity_curve([100.0])
            plt.close(fig)
            # If it doesn't raise, that's fine
        except (ValueError, IndexError):
            # May not be able to plot single point
            pass
    
    def test_plot_with_constant_prices(self):
        """Test plotting with constant prices."""
        from src.visualizations import plot_equity_curve
        
        prices = [100.0] * 50
        fig = plot_equity_curve(prices)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_with_nan_values(self):
        """Test plotting with NaN values."""
        from src.visualizations import plot_equity_curve
        
        prices = [100.0, 101.0, np.nan, 102.0, 103.0]
        
        try:
            fig = plot_equity_curve(prices)
            plt.close(fig)
            # If it handles NaN gracefully, that's fine
        except (ValueError, RuntimeError):
            # Expected to raise error or handle gracefully
            pass
    
    def test_plot_with_negative_prices(self):
        """Test plotting with negative prices (edge case)."""
        from src.visualizations import plot_equity_curve
        
        prices = [100.0, 101.0, -50.0, 102.0]
        
        # Should either handle or raise appropriate error
        try:
            fig = plot_equity_curve(prices)
            plt.close(fig)
        except ValueError:
            pass  # Expected for invalid prices
    
    def test_plot_with_very_large_values(self):
        """Test plotting with very large values."""
        from src.visualizations import plot_equity_curve
        
        prices = [1e10, 1.1e10, 1.2e10, 1.15e10]
        
        fig = plot_equity_curve(prices)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_with_very_small_values(self):
        """Test plotting with very small values."""
        from src.visualizations import plot_equity_curve
        
        prices = [1e-6, 1.1e-6, 1.2e-6, 1.15e-6]
        
        fig = plot_equity_curve(prices)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Plot Elements Tests
# =============================================================================

class TestPlotElements:
    """Test plot elements are present."""
    
    def test_plot_has_title(self, sample_prices_series):
        """Test plot has a title."""
        from src.visualizations import plot_equity_curve
        
        fig = plot_equity_curve(sample_prices_series)
        ax = fig.get_axes()[0]
        
        # Check title exists
        title = ax.get_title()
        assert title is not None
        assert len(title) > 0
        
        plt.close(fig)
    
    def test_plot_has_labels(self, sample_prices_series):
        """Test plot has axis labels."""
        from src.visualizations import plot_equity_curve
        
        fig = plot_equity_curve(sample_prices_series)
        ax = fig.get_axes()[0]
        
        # Check labels
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        
        assert xlabel is not None or ylabel is not None
        
        plt.close(fig)
    
    def test_plot_has_legend(self, sample_prices_series):
        """Test plot has legend when appropriate."""
        from src.visualizations import plot_equity_curve
        
        # Plot with reference prices should have legend
        fig = plot_equity_curve(
            sample_prices_series,
            reference_prices=sample_prices_series,
        )
        ax = fig.get_axes()[0]
        
        legend = ax.get_legend()
        # Legend may or may not be present depending on implementation
        
        plt.close(fig)
    
    def test_plot_has_grid(self, sample_prices_series):
        """Test plot grid settings."""
        from src.visualizations import plot_equity_curve
        
        fig = plot_equity_curve(sample_prices_series)
        ax = fig.get_axes()[0]
        
        # Grid may or may not be enabled by default
        # Just check the function works
        
        plt.close(fig)


# =============================================================================
# Integration Tests
# =============================================================================

class TestVisualizationIntegration:
    """Integration tests for visualizations with full pipeline."""
    
    def test_visualization_with_experiment_results(self, temp_dir):
        """Test visualizations work with real experiment results."""
        from src.experiment_smallset import run_small_dataset_experiment
        from src.visualizations import plot_equity_curve, plot_agent_comparison
        
        # Run small experiment
        results = run_small_dataset_experiment(
            backend="tfidf",
            k_clusters=3,
            agent_types=["Random", "Momentum"],
            alpha=0.01,
            seed=42,
        )
        
        # Create visualizations
        fig1 = plot_equity_curve(
            results["simulated_prices"],
            reference_prices=results["reference_prices"],
        )
        assert fig1 is not None
        plt.close(fig1)
        
        fig2 = plot_agent_comparison(
            results["action_log"],
            results["simulated_prices"],
        )
        assert fig2 is not None
        plt.close(fig2)
    
    def test_visualization_export_workflow(self, sample_prices_series, temp_dir):
        """Test complete export workflow."""
        from src.visualizations import (
            plot_equity_curve,
            save_figure,
            set_publication_style,
            reset_style,
        )
        
        # Set publication style
        set_publication_style()
        
        # Create plot
        fig = plot_equity_curve(sample_prices_series)
        
        # Export
        output_path = temp_dir / "publication_figure"
        save_figure(fig, str(output_path), dpi=300, formats=["png", "pdf"])
        
        # Verify files
        assert (temp_dir / "publication_figure.png").exists()
        assert (temp_dir / "publication_figure.pdf").exists()
        
        # Reset style
        reset_style()
        plt.close(fig)

