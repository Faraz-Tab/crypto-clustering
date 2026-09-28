"""Render static PNG figures for the README, since hvPlot output does not display on GitHub."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crypto_clustering import cluster, elbow, load_data, reduce_pca, scale, silhouette

OUT = ROOT / "images"
K = 4
OUTLIERS = ["ethlend", "celsius-degree-token"]
COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def line_pair(ax, original, pca, x, y, title, ylabel):
    ax.plot(original[x], original[y], marker="o", label="Original (7 features)")
    ax.plot(pca[x], pca[y], marker="s", label="PCA (3 components)")
    ax.axvline(K, color="grey", linestyle="--", linewidth=1)
    ax.set(title=title, xlabel="k", ylabel=ylabel, xticks=list(original[x]))
    ax.legend()


def scatter(ax, df, labels, x, y, title):
    for c in sorted(labels.unique()):
        members = df[labels == c]
        ax.scatter(members[x], members[y], s=40, color=COLOURS[c % len(COLOURS)], label=f"Cluster {c} ({len(members)})")
    for coin in OUTLIERS:
        ax.annotate(coin, (df.loc[coin, x], df.loc[coin, y]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set(title=title, xlabel=x, ylabel=y)
    ax.legend(fontsize=8)


def main():
    OUT.mkdir(exist_ok=True)
    scaled = scale(load_data())
    pca_df, _ = reduce_pca(scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    line_pair(axes[0], elbow(scaled), elbow(pca_df), "k", "inertia", "Elbow curve", "Inertia")
    line_pair(axes[1], silhouette(scaled), silhouette(pca_df), "k", "silhouette", "Silhouette score", "Silhouette")
    save(fig, "elbow_silhouette.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter(axes[0], scaled, cluster(scaled, K), "price_change_percentage_24h", "price_change_percentage_7d",
            "Original features (scaled), 2 of 7 shown")
    scatter(axes[1], pca_df, cluster(pca_df, K), "PC1", "PC2", "PCA components")
    save(fig, "clusters.png")


if __name__ == "__main__":
    main()
