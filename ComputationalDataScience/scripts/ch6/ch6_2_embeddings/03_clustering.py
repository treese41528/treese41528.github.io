"""
STAT 418 · Chapter 6.2 — Embeddings & Feature Extraction
Script 03: Clustering embeddings — K-means on the embedding vectors, then a 2D
PCA projection for visualization.

Embeddings are just high-dimensional feature vectors, so the usual scikit-learn
tools apply directly. K-means discovers the natural topic groupings with no
labels; PCA projects the 3072-d vectors to 2D so we can see the structure.

Note: K-means cluster ids are arbitrary (cluster 0 is not necessarily "Sports"),
so we label clusters by id and let the printed members reveal the topic.

Prereqs / run: see 01_generate.py. (Also needs scikit-learn and matplotlib.)
Saves: embedding_clusters.png
"""
from __future__ import annotations

import os
import sys

import numpy as np

from genai_studio import GenAIStudio

EMBED_MODEL = "llama3.2:latest"

DOCUMENTS = [
    # Sports
    "The quarterback threw a touchdown pass in the fourth quarter.",
    "The tennis player won the Grand Slam tournament.",
    "The soccer team scored three goals in the first half.",
    "The marathon runner set a new personal record.",
    "The basketball team won the championship game.",
    # Technology
    "The new smartphone features a faster processor.",
    "Cloud computing enables scalable data storage.",
    "The software update fixed several security vulnerabilities.",
    "Artificial intelligence is transforming healthcare.",
    "The database query returned results in milliseconds.",
    # Cooking
    "Preheat the oven to 375 degrees Fahrenheit.",
    "The sourdough starter needs to ferment for 12 hours.",
    "Season the chicken with salt, pepper, and rosemary.",
    "The chocolate cake needs to cool before frosting.",
    "Dice the onions and sauté them in olive oil.",
]


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set — see ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(EMBED_MODEL)
    return ai


def main() -> None:
    from sklearn.cluster import KMeans

    ai = make_client()
    X = np.array(ai.embed(DOCUMENTS))

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    for cluster_id in range(3):
        print(f"\nCluster {cluster_id}:")
        for doc, lab in zip(DOCUMENTS, labels):
            if lab == cluster_id:
                print(f"  - {doc[:60]}...")

    # --- 2D PCA projection for visualization ---
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    colors = ["#2E86AB", "#2D936C", "#F18F01"]
    plt.figure(figsize=(8, 6))
    for cluster_id in range(3):
        mask = labels == cluster_id
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[cluster_id], s=80,
                    alpha=0.8, label=f"Cluster {cluster_id}", edgecolors="white")
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("Document Clusters in Embedding Space")
    plt.legend()
    plt.tight_layout()
    out = "embedding_clusters.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
