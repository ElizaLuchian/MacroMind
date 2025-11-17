import numpy as np

from src.embedder import NewsEmbedder


def test_tfidf_backend_shapes_and_normalization():
    embedder = NewsEmbedder(backend="tfidf", random_state=0)
    texts = ["alpha beta", "gamma delta"]
    embeddings = embedder.encode_news(texts)
    assert embeddings.shape[0] == 2
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)




