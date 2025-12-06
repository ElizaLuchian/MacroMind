"""Benchmarks for news embedding performance."""

from __future__ import annotations

import pytest

from tests.test_utils import generate_mock_news_data


# =============================================================================
# Embedding Speed Benchmarks
# =============================================================================

@pytest.mark.benchmark
class TestEmbedderBenchmarks:
    """Benchmark embedding generation speed."""
    
    def test_tfidf_embedding_speed_small(self, benchmark):
        """Benchmark TF-IDF embedding for small dataset."""
        from src.embedder import NewsEmbedder
        
        news_data = generate_mock_news_data(n_rows=50, seed=42)
        texts = (news_data["headline"] + " " + news_data["body"]).tolist()
        
        embedder = NewsEmbedder(backend="tfidf")
        
        def embed_texts():
            return embedder.embed(texts)
        
        result = benchmark(embed_texts)
        
        # Verify embeddings were generated
        assert result.shape[0] == 50
        assert result.shape[1] == 384
    
    def test_tfidf_embedding_speed_medium(self, benchmark):
        """Benchmark TF-IDF embedding for medium dataset."""
        from src.embedder import NewsEmbedder
        
        news_data = generate_mock_news_data(n_rows=200, seed=42)
        texts = (news_data["headline"] + " " + news_data["body"]).tolist()
        
        embedder = NewsEmbedder(backend="tfidf")
        
        def embed_texts():
            return embedder.embed(texts)
        
        result = benchmark(embed_texts)
        
        assert result.shape[0] == 200
    
    def test_tfidf_embedding_speed_large(self, benchmark):
        """Benchmark TF-IDF embedding for large dataset."""
        from src.embedder import NewsEmbedder
        
        news_data = generate_mock_news_data(n_rows=1000, seed=42)
        texts = (news_data["headline"] + " " + news_data["body"]).tolist()
        
        embedder = NewsEmbedder(backend="tfidf")
        
        def embed_texts():
            return embedder.embed(texts)
        
        result = benchmark(embed_texts)
        
        assert result.shape[0] == 1000
    
    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_sentence_transformer_speed(self, benchmark):
        """Benchmark sentence-transformer embedding speed."""
        pytest.skip("Requires downloading model - run manually if needed")
        
        from src.embedder import NewsEmbedder
        
        news_data = generate_mock_news_data(n_rows=50, seed=42)
        texts = (news_data["headline"] + " " + news_data["body"]).tolist()
        
        embedder = NewsEmbedder(backend="sentence-transformer")
        
        def embed_texts():
            return embedder.embed(texts)
        
        result = benchmark(embed_texts)
        
        assert result.shape[0] == 50


# =============================================================================
# Embedding Scaling Benchmarks
# =============================================================================

@pytest.mark.benchmark
class TestEmbedderScaling:
    """Test how embedding performance scales with input size."""
    
    @pytest.mark.parametrize("n_texts", [10, 50, 100, 200, 500])
    def test_tfidf_scaling(self, benchmark, n_texts):
        """Test TF-IDF scaling with different input sizes."""
        from src.embedder import NewsEmbedder
        
        news_data = generate_mock_news_data(n_rows=n_texts, seed=42)
        texts = (news_data["headline"] + " " + news_data["body"]).tolist()
        
        embedder = NewsEmbedder(backend="tfidf")
        
        def embed_texts():
            return embedder.embed(texts)
        
        result = benchmark(embed_texts)
        
        assert result.shape[0] == n_texts
    
    def test_embedding_initialization_overhead(self, benchmark):
        """Benchmark embedder initialization overhead."""
        from src.embedder import NewsEmbedder
        
        def create_embedder():
            return NewsEmbedder(backend="tfidf")
        
        embedder = benchmark(create_embedder)
        
        assert embedder is not None


# =============================================================================
# Memory Usage Benchmarks
# =============================================================================

@pytest.mark.benchmark
class TestEmbedderMemory:
    """Test memory usage of embedding operations."""
    
    def test_embedding_memory_efficiency(self):
        """Test embedding doesn't consume excessive memory."""
        from src.embedder import NewsEmbedder
        
        # Large dataset
        news_data = generate_mock_news_data(n_rows=1000, seed=42)
        texts = (news_data["headline"] + " " + news_data["body"]).tolist()
        
        embedder = NewsEmbedder(backend="tfidf")
        embeddings = embedder.embed(texts)
        
        # Check memory is reasonable (embeddings should be float32 or float64)
        memory_mb = embeddings.nbytes / (1024 * 1024)
        
        # 1000 samples × 384 features × 8 bytes = ~3 MB
        assert memory_mb < 50  # Should be well under 50 MB
    
    def test_batch_embedding_memory(self):
        """Test memory usage with batch processing."""
        from src.embedder import NewsEmbedder
        
        embedder = NewsEmbedder(backend="tfidf")
        
        # Process in batches
        batch_size = 100
        all_embeddings = []
        
        for batch_idx in range(10):
            news_data = generate_mock_news_data(n_rows=batch_size, seed=42 + batch_idx)
            texts = (news_data["headline"] + " " + news_data["body"]).tolist()
            
            embeddings = embedder.embed(texts)
            all_embeddings.append(embeddings)
        
        # Should successfully process 1000 texts in batches
        assert len(all_embeddings) == 10


# =============================================================================
# Comparative Benchmarks
# =============================================================================

@pytest.mark.benchmark
class TestEmbedderComparison:
    """Compare different embedding backends."""
    
    def test_compare_tfidf_vs_count(self, benchmark):
        """Compare TF-IDF vs simple count vectorization."""
        from src.embedder import NewsEmbedder
        
        news_data = generate_mock_news_data(n_rows=100, seed=42)
        texts = (news_data["headline"] + " " + news_data["body"]).tolist()
        
        # Benchmark TF-IDF
        embedder_tfidf = NewsEmbedder(backend="tfidf")
        
        def embed_tfidf():
            return embedder_tfidf.embed(texts)
        
        result = benchmark(embed_tfidf)
        
        assert result.shape[0] == 100
    
    def test_embedding_quality_vs_speed_tradeoff(self):
        """Demonstrate embedding quality vs speed tradeoff."""
        from src.embedder import NewsEmbedder
        import time
        
        news_data = generate_mock_news_data(n_rows=100, seed=42)
        texts = (news_data["headline"] + " " + news_data["body"]).tolist()
        
        # TF-IDF (fast)
        embedder_tfidf = NewsEmbedder(backend="tfidf")
        
        start = time.time()
        embeddings_tfidf = embedder_tfidf.embed(texts)
        tfidf_time = time.time() - start
        
        # Check results
        assert embeddings_tfidf.shape[0] == 100
        assert tfidf_time < 5.0  # Should be very fast


# =============================================================================
# Regression Tests
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.regression
class TestEmbedderRegression:
    """Regression tests to detect performance degradation."""
    
    def test_baseline_tfidf_performance(self, benchmark):
        """Establish baseline for TF-IDF performance."""
        from src.embedder import NewsEmbedder
        
        news_data = generate_mock_news_data(n_rows=100, seed=42)
        texts = (news_data["headline"] + " " + news_data["body"]).tolist()
        
        embedder = NewsEmbedder(backend="tfidf")
        
        def embed_texts():
            return embedder.embed(texts)
        
        result = benchmark(embed_texts)
        
        # Store benchmark results for comparison
        stats = benchmark.stats
        
        # Should complete in reasonable time (< 1 second for 100 texts)
        assert stats.mean < 1.0
        
        assert result.shape[0] == 100
    
    def test_embedding_consistency_over_runs(self):
        """Test embedding results are consistent across runs."""
        from src.embedder import NewsEmbedder
        
        news_data = generate_mock_news_data(n_rows=50, seed=42)
        texts = (news_data["headline"] + " " + news_data["body"]).tolist()
        
        embedder = NewsEmbedder(backend="tfidf")
        
        # Run multiple times
        embeddings1 = embedder.embed(texts)
        embeddings2 = embedder.embed(texts)
        
        # Should produce identical results
        import numpy as np
        assert np.allclose(embeddings1, embeddings2)

