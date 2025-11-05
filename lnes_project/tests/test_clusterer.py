import numpy as np

from src.clusterer import cluster_embeddings


def test_cluster_embeddings_returns_valid_labels():
    embeddings = np.vstack([np.ones((5, 3)), np.zeros((5, 3))])
    result = cluster_embeddings(embeddings, k=2, random_state=1)
    assert set(result.labels) == {0, 1}
    assert result.model.cluster_centers_.shape == (2, 3)


