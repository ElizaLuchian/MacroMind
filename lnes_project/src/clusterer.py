"""Clustering utilities for latent news event detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusterResult:
    labels: np.ndarray
    model: KMeans
    inertia: float
    silhouette: Optional[float]


def cluster_embeddings(
    embeddings: np.ndarray,
    k: int = 3,
    random_state: int = 42,
    max_iter: int = 300,
) -> ClusterResult:
    """Run KMeans on the embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if not 2 <= k <= 20:
        raise ValueError("k must be between 2 and 20 for stability.")

    model = KMeans(n_clusters=k, random_state=random_state, max_iter=max_iter, n_init="auto")
    labels = model.fit_predict(embeddings)
    score = None
    if len(np.unique(labels)) > 1:
        score = float(silhouette_score(embeddings, labels))
    return ClusterResult(labels=labels, model=model, inertia=float(model.inertia_), silhouette=score)


def assign_clusters(embeddings: np.ndarray, model: KMeans) -> np.ndarray:
    """Assign new embeddings to existing clusters."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    return model.predict(embeddings)


def plot_cluster_distribution(labels: Iterable[int], ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Visualize cluster membership counts."""
    data = list(labels)
    if not data:
        raise ValueError("No labels provided for plotting.")

    counts = np.bincount(data)
    axis = ax or plt.gca()
    axis.bar(range(len(counts)), counts, color="#4C72B0")
    axis.set_xlabel("Cluster ID")
    axis.set_ylabel("Frequency")
    axis.set_title("Latent Event Cluster Distribution")
    return axis


