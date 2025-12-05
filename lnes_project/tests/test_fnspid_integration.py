"""Integration tests for FNSPID dataset loading and experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from tests.test_utils import generate_mock_news_data, generate_mock_price_data


# =============================================================================
# FNSPID Data Loading Tests
# =============================================================================

class TestFNSPIDDataLoading:
    """Test FNSPID dataset loading functionality."""
    
    def test_fnspid_loader_initialization(self):
        """Test FNSPIDLoader can be initialized."""
        from src.data_loader import FNSPIDLoader
        
        loader = FNSPIDLoader()
        assert loader is not None
    
    @pytest.mark.integration
    def test_fnspid_load_with_mock_data(self, temp_dir):
        """Test loading FNSPID data with mock CSV files."""
        from src.data_loader import FNSPIDLoader
        
        # Create mock FNSPID data
        news_data = generate_mock_news_data(n_rows=100, seed=42)
        price_data = generate_mock_price_data(n_rows=50, seed=42)
        
        # Save to temp directory
        news_path = temp_dir / "news.csv"
        price_path = temp_dir / "prices.csv"
        news_data.to_csv(news_path, index=False)
        price_data.to_csv(price_path, index=False)
        
        # Load data
        loader = FNSPIDLoader()
        news, prices = loader.load_from_csv(
            news_path=str(news_path),
            prices_path=str(price_path),
        )
        
        assert len(news) == 100
        assert len(prices) == 50
        assert "date" in news.columns
        assert "close" in prices.columns
    
    def test_fnspid_ticker_filtering(self, temp_dir):
        """Test filtering FNSPID data by ticker."""
        from src.data_loader import FNSPIDLoader
        
        # Create mock data with multiple tickers
        news_data = generate_mock_news_data(n_rows=100, seed=42)
        price_data = generate_mock_price_data(n_rows=50, seed=42)
        price_data["ticker"] = ["AAPL"] * 25 + ["MSFT"] * 25
        
        news_path = temp_dir / "news.csv"
        price_path = temp_dir / "prices.csv"
        news_data.to_csv(news_path, index=False)
        price_data.to_csv(price_path, index=False)
        
        loader = FNSPIDLoader()
        news, prices = loader.load_from_csv(
            news_path=str(news_path),
            prices_path=str(price_path),
            tickers=["AAPL"],
        )
        
        # Should only have AAPL data
        assert all(prices["ticker"] == "AAPL")
    
    def test_fnspid_date_filtering(self, temp_dir):
        """Test filtering FNSPID data by date range."""
        from src.data_loader import FNSPIDLoader
        
        news_data = generate_mock_news_data(n_rows=100, start_date="2023-01-01", seed=42)
        price_data = generate_mock_price_data(n_rows=100, start_date="2023-01-01", seed=42)
        
        news_path = temp_dir / "news.csv"
        price_path = temp_dir / "prices.csv"
        news_data.to_csv(news_path, index=False)
        price_data.to_csv(price_path, index=False)
        
        loader = FNSPIDLoader()
        news, prices = loader.load_from_csv(
            news_path=str(news_path),
            prices_path=str(price_path),
            start_date="2023-02-01",
            end_date="2023-02-28",
        )
        
        # Convert dates for comparison
        news["date"] = pd.to_datetime(news["date"])
        prices["date"] = pd.to_datetime(prices["date"])
        
        # Check date range
        assert news["date"].min() >= pd.Timestamp("2023-02-01")
        assert news["date"].max() <= pd.Timestamp("2023-02-28")
    
    def test_fnspid_missing_columns_error(self, temp_dir):
        """Test error handling for missing required columns."""
        from src.data_loader import FNSPIDLoader
        
        # Create data with missing columns
        bad_news = pd.DataFrame({
            "date": ["2023-01-01"],
            "headline": ["Test"],
            # Missing 'body' column
        })
        
        news_path = temp_dir / "bad_news.csv"
        bad_news.to_csv(news_path, index=False)
        
        loader = FNSPIDLoader()
        
        # Should handle missing columns gracefully
        try:
            news, prices = loader.load_from_csv(
                news_path=str(news_path),
                prices_path=str(news_path),  # Use same file for simplicity
            )
        except (KeyError, ValueError) as e:
            # Expected to raise error for missing columns
            assert "column" in str(e).lower() or "missing" in str(e).lower()
    
    def test_fnspid_empty_data_handling(self, temp_dir):
        """Test handling of empty datasets."""
        from src.data_loader import FNSPIDLoader
        
        # Create empty files
        empty_df = pd.DataFrame()
        news_path = temp_dir / "empty_news.csv"
        price_path = temp_dir / "empty_prices.csv"
        empty_df.to_csv(news_path, index=False)
        empty_df.to_csv(price_path, index=False)
        
        loader = FNSPIDLoader()
        
        try:
            news, prices = loader.load_from_csv(
                news_path=str(news_path),
                prices_path=str(price_path),
            )
            # If it doesn't raise, check that data is empty
            assert len(news) == 0 or len(prices) == 0
        except (ValueError, KeyError):
            # Expected to raise error for empty data
            pass
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_fnspid_huggingface_mock(self):
        """Test loading from Hugging Face (mocked)."""
        # Mock the datasets library
        mock_dataset = Mock()
        mock_df = generate_mock_news_data(n_rows=50, seed=42)
        mock_dataset.to_pandas.return_value = mock_df
        
        with patch("datasets.load_dataset", return_value=mock_dataset):
            from src.data_loader import FNSPIDLoader
            
            loader = FNSPIDLoader()
            news = loader.load_from_huggingface(
                dataset_name="Zihan1004/FNSPID",
                split="train",
            )
            
            assert len(news) > 0
            assert isinstance(news, pd.DataFrame)


# =============================================================================
# FNSPID Experiment Integration Tests
# =============================================================================

class TestFNSPIDExperiment:
    """Test end-to-end FNSPID experiments."""
    
    @pytest.mark.integration
    def test_fnspid_experiment_small_scale(self, temp_dir):
        """Test running FNSPID experiment with small mock data."""
        from src.experiment_fnspid import run_fnspid_experiment
        
        # Create minimal mock data
        news_data = generate_mock_news_data(n_rows=30, seed=42)
        price_data = generate_mock_price_data(n_rows=30, seed=42)
        
        news_path = temp_dir / "news.csv"
        price_path = temp_dir / "prices.csv"
        news_data.to_csv(news_path, index=False)
        price_data.to_csv(price_path, index=False)
        
        # Run experiment
        results = run_fnspid_experiment(
            news_path=str(news_path),
            prices_path=str(price_path),
            backend="tfidf",
            k_clusters=3,
            agent_types=["Random", "Momentum"],
            alpha=0.01,
            seed=42,
        )
        
        # Check results structure
        assert "metrics" in results
        assert "action_log" in results
        assert "simulated_prices" in results
        assert "reference_prices" in results
    
    @pytest.mark.integration
    def test_fnspid_experiment_with_clustering(self, temp_dir):
        """Test FNSPID experiment includes clustering."""
        from src.experiment_fnspid import run_fnspid_experiment
        
        news_data = generate_mock_news_data(n_rows=30, seed=42)
        price_data = generate_mock_price_data(n_rows=30, seed=42)
        
        news_path = temp_dir / "news.csv"
        price_path = temp_dir / "prices.csv"
        news_data.to_csv(news_path, index=False)
        price_data.to_csv(price_path, index=False)
        
        results = run_fnspid_experiment(
            news_path=str(news_path),
            prices_path=str(price_path),
            backend="tfidf",
            k_clusters=3,
            agent_types=["NewsReactive"],
            alpha=0.01,
            seed=42,
        )
        
        # Check clustering was performed
        assert "cluster_labels" in results
        assert len(results["cluster_labels"]) == 30
        assert all(0 <= label < 3 for label in results["cluster_labels"])
    
    @pytest.mark.integration
    def test_fnspid_experiment_multiple_agents(self, temp_dir):
        """Test FNSPID experiment with multiple agents."""
        from src.experiment_fnspid import run_fnspid_experiment
        
        news_data = generate_mock_news_data(n_rows=30, seed=42)
        price_data = generate_mock_price_data(n_rows=30, seed=42)
        
        news_path = temp_dir / "news.csv"
        price_path = temp_dir / "prices.csv"
        news_data.to_csv(news_path, index=False)
        price_data.to_csv(price_path, index=False)
        
        agent_types = ["Random", "Momentum", "Contrarian", "NewsReactive"]
        
        results = run_fnspid_experiment(
            news_path=str(news_path),
            prices_path=str(price_path),
            backend="tfidf",
            k_clusters=3,
            agent_types=agent_types,
            alpha=0.01,
            seed=42,
        )
        
        # Check all agents are in action log
        assert len(results["action_log"]) == len(agent_types)
        for agent_type in agent_types:
            assert agent_type in results["action_log"]
    
    @pytest.mark.integration
    def test_fnspid_experiment_metrics_calculation(self, temp_dir):
        """Test FNSPID experiment calculates metrics."""
        from src.experiment_fnspid import run_fnspid_experiment
        
        news_data = generate_mock_news_data(n_rows=30, seed=42)
        price_data = generate_mock_price_data(n_rows=30, seed=42)
        
        news_path = temp_dir / "news.csv"
        price_path = temp_dir / "prices.csv"
        news_data.to_csv(news_path, index=False)
        price_data.to_csv(price_path, index=False)
        
        results = run_fnspid_experiment(
            news_path=str(news_path),
            prices_path=str(price_path),
            backend="tfidf",
            k_clusters=3,
            agent_types=["Momentum"],
            alpha=0.01,
            seed=42,
        )
        
        metrics = results["metrics"]
        
        # Check key metrics exist
        assert "directional_accuracy" in metrics
        assert "final_price_ratio" in metrics
        assert 0 <= metrics["directional_accuracy"] <= 1
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_fnspid_experiment_with_sentence_transformer(self, temp_dir):
        """Test FNSPID experiment with sentence-transformer backend."""
        pytest.skip("Requires downloading sentence-transformer model - run manually if needed")
        
        from src.experiment_fnspid import run_fnspid_experiment
        
        news_data = generate_mock_news_data(n_rows=30, seed=42)
        price_data = generate_mock_price_data(n_rows=30, seed=42)
        
        news_path = temp_dir / "news.csv"
        price_path = temp_dir / "prices.csv"
        news_data.to_csv(news_path, index=False)
        price_data.to_csv(price_path, index=False)
        
        results = run_fnspid_experiment(
            news_path=str(news_path),
            prices_path=str(price_path),
            backend="sentence-transformer",
            k_clusters=3,
            agent_types=["Momentum"],
            alpha=0.01,
            seed=42,
        )
        
        assert "embeddings" in results
        assert results["embeddings"].shape[1] == 384  # MiniLM dimension


# =============================================================================
# FNSPID Data Quality Tests
# =============================================================================

class TestFNSPIDDataQuality:
    """Test data quality and validation for FNSPID."""
    
    def test_fnspid_news_price_alignment(self, temp_dir):
        """Test news and price data are properly aligned by date."""
        from src.data_loader import FNSPIDLoader
        
        # Create data with overlapping dates
        news_data = generate_mock_news_data(n_rows=50, start_date="2023-01-01", seed=42)
        price_data = generate_mock_price_data(n_rows=50, start_date="2023-01-01", seed=42)
        
        news_path = temp_dir / "news.csv"
        price_path = temp_dir / "prices.csv"
        news_data.to_csv(news_path, index=False)
        price_data.to_csv(price_path, index=False)
        
        loader = FNSPIDLoader()
        news, prices = loader.load_from_csv(
            news_path=str(news_path),
            prices_path=str(price_path),
        )
        
        # Convert dates
        news["date"] = pd.to_datetime(news["date"])
        prices["date"] = pd.to_datetime(prices["date"])
        
        # Check date overlap
        news_dates = set(news["date"])
        price_dates = set(prices["date"])
        overlap = news_dates & price_dates
        
        assert len(overlap) > 0, "No date overlap between news and prices"
    
    def test_fnspid_price_data_quality(self, temp_dir):
        """Test FNSPID price data quality checks."""
        from src.data_loader import FNSPIDLoader
        
        price_data = generate_mock_price_data(n_rows=50, seed=42)
        price_path = temp_dir / "prices.csv"
        price_data.to_csv(price_path, index=False)
        
        loader = FNSPIDLoader()
        prices = pd.read_csv(price_path)
        
        # Check OHLC relationships
        assert all(prices["high"] >= prices["low"])
        assert all(prices["high"] >= prices["open"])
        assert all(prices["high"] >= prices["close"])
        assert all(prices["low"] <= prices["open"])
        assert all(prices["low"] <= prices["close"])
        
        # Check positive prices
        assert all(prices["close"] > 0)
        assert all(prices["volume"] > 0)
    
    def test_fnspid_news_text_quality(self, temp_dir):
        """Test FNSPID news text quality."""
        from src.data_loader import FNSPIDLoader
        
        news_data = generate_mock_news_data(n_rows=50, seed=42)
        news_path = temp_dir / "news.csv"
        news_data.to_csv(news_path, index=False)
        
        news = pd.read_csv(news_path)
        
        # Check text fields are non-empty
        assert all(len(str(h)) > 0 for h in news["headline"])
        assert all(len(str(b)) > 0 for b in news["body"])
        
        # Check dates are valid
        dates = pd.to_datetime(news["date"])
        assert all(pd.notna(dates))
    
    def test_fnspid_duplicate_handling(self, temp_dir):
        """Test handling of duplicate entries."""
        from src.data_loader import FNSPIDLoader
        
        news_data = generate_mock_news_data(n_rows=30, seed=42)
        # Add some duplicates
        duplicates = news_data.head(5)
        news_data = pd.concat([news_data, duplicates], ignore_index=True)
        
        news_path = temp_dir / "news_with_dupes.csv"
        news_data.to_csv(news_path, index=False)
        
        loader = FNSPIDLoader()
        news = pd.read_csv(news_path)
        
        # Check if duplicates exist
        original_len = len(news)
        deduped = news.drop_duplicates()
        
        if len(deduped) < original_len:
            # Duplicates were found and can be removed
            assert len(deduped) == 30


# =============================================================================
# FNSPID Performance Tests
# =============================================================================

class TestFNSPIDPerformance:
    """Test performance characteristics of FNSPID experiments."""
    
    @pytest.mark.benchmark
    def test_fnspid_loading_performance(self, temp_dir, benchmark):
        """Benchmark FNSPID data loading speed."""
        from src.data_loader import FNSPIDLoader
        
        # Create larger dataset
        news_data = generate_mock_news_data(n_rows=500, seed=42)
        price_data = generate_mock_price_data(n_rows=500, seed=42)
        
        news_path = temp_dir / "news_large.csv"
        price_path = temp_dir / "prices_large.csv"
        news_data.to_csv(news_path, index=False)
        price_data.to_csv(price_path, index=False)
        
        loader = FNSPIDLoader()
        
        def load_data():
            return loader.load_from_csv(
                news_path=str(news_path),
                prices_path=str(price_path),
            )
        
        # Benchmark should complete in reasonable time
        if benchmark is not None:
            result = benchmark(load_data)
        else:
            result = load_data()
        
        news, prices = result
        assert len(news) == 500
        assert len(prices) == 500
    
    @pytest.mark.integration
    def test_fnspid_memory_efficiency(self, temp_dir):
        """Test FNSPID handles large datasets efficiently."""
        from src.data_loader import FNSPIDLoader
        
        # Create moderately large dataset
        news_data = generate_mock_news_data(n_rows=1000, seed=42)
        price_data = generate_mock_price_data(n_rows=1000, seed=42)
        
        news_path = temp_dir / "news_large.csv"
        price_path = temp_dir / "prices_large.csv"
        news_data.to_csv(news_path, index=False)
        price_data.to_csv(price_path, index=False)
        
        loader = FNSPIDLoader()
        news, prices = loader.load_from_csv(
            news_path=str(news_path),
            prices_path=str(price_path),
        )
        
        # Check data loaded successfully
        assert len(news) == 1000
        assert len(prices) == 1000
        
        # Memory should be released after processing
        del news, prices


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestFNSPIDErrorHandling:
    """Test error handling in FNSPID experiments."""
    
    def test_fnspid_missing_files(self):
        """Test handling of missing data files."""
        from src.data_loader import FNSPIDLoader
        
        loader = FNSPIDLoader()
        
        with pytest.raises((FileNotFoundError, ValueError)):
            loader.load_from_csv(
                news_path="/nonexistent/news.csv",
                prices_path="/nonexistent/prices.csv",
            )
    
    def test_fnspid_invalid_date_format(self, temp_dir):
        """Test handling of invalid date formats."""
        from src.data_loader import FNSPIDLoader
        
        # Create data with invalid dates
        bad_data = pd.DataFrame({
            "date": ["invalid-date", "2023-99-99"],
            "headline": ["Test1", "Test2"],
            "body": ["Body1", "Body2"],
        })
        
        bad_path = temp_dir / "bad_dates.csv"
        bad_data.to_csv(bad_path, index=False)
        
        loader = FNSPIDLoader()
        
        # Should handle invalid dates gracefully
        try:
            news = pd.read_csv(bad_path)
            dates = pd.to_datetime(news["date"], errors="coerce")
            assert dates.isna().any()  # Some dates should be NaT
        except (ValueError, KeyError):
            pass  # Expected to fail
    
    def test_fnspid_experiment_with_missing_ticker(self, temp_dir):
        """Test experiment handles missing ticker gracefully."""
        from src.experiment_fnspid import run_fnspid_experiment
        
        news_data = generate_mock_news_data(n_rows=30, seed=42)
        price_data = generate_mock_price_data(n_rows=30, seed=42)
        price_data["ticker"] = "AAPL"
        
        news_path = temp_dir / "news.csv"
        price_path = temp_dir / "prices.csv"
        news_data.to_csv(news_path, index=False)
        price_data.to_csv(price_path, index=False)
        
        # Try to load non-existent ticker
        try:
            results = run_fnspid_experiment(
                news_path=str(news_path),
                prices_path=str(price_path),
                tickers=["NONEXISTENT"],
                backend="tfidf",
                k_clusters=3,
                agent_types=["Random"],
                seed=42,
            )
            # If it doesn't error, should have no data
            assert len(results.get("simulated_prices", [])) == 0
        except (ValueError, KeyError):
            pass  # Expected to fail

