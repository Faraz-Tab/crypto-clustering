import sys
from pathlib import Path

import pytest
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crypto_clustering import cluster, load_data, reduce_pca, scale, silhouette


@pytest.fixture(scope="module")
def scaled():
    return scale(load_data())


def test_data_shape(scaled):
    assert scaled.shape == (41, 7)


def test_pca_explains_most_variance(scaled):
    _, ratio = reduce_pca(scaled)
    assert ratio.sum() == pytest.approx(0.895, abs=0.001)


def test_clustering_is_reproducible(scaled):
    assert (cluster(scaled, 4) == cluster(scaled, 4)).all()


def test_pca_reproduces_original_grouping(scaled):
    pca_df, _ = reduce_pca(scaled)
    assert adjusted_rand_score(cluster(scaled, 4), cluster(pca_df, 4)) == pytest.approx(1.0)


def test_k4_isolates_two_outliers(scaled):
    labels = cluster(scaled, 4)
    sizes = labels.value_counts()
    singletons = {labels[labels == c].index[0] for c in sizes[sizes == 1].index}
    assert singletons == {"ethlend", "celsius-degree-token"}


def test_silhouette_drops_after_k3(scaled):
    scores = silhouette(scaled, range(2, 6)).set_index("k")["silhouette"]
    assert scores[3] > scores[4] * 2
