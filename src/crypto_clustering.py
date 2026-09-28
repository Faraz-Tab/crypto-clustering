"""Scaling, PCA and K-means utilities for clustering cryptocurrencies by price-change behaviour."""
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "Resources" / "crypto_market_data.csv"
RANDOM_STATE = 0
N_INIT = 10  # several initializations make K-means results stable across runs and library versions


def load_data(path=DATA_PATH):
    return pd.read_csv(path, index_col="coin_id")


def scale(df):
    return pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns, index=df.index)


def reduce_pca(df_scaled, n_components=3):
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    components = pca.fit_transform(df_scaled)
    columns = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(components, columns=columns, index=df_scaled.index), pca.explained_variance_ratio_


def kmeans(n_clusters):
    return KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=N_INIT)


def cluster(df, n_clusters):
    return pd.Series(kmeans(n_clusters).fit_predict(df), index=df.index, name="cluster")


def elbow(df, k_values=range(1, 11)):
    return pd.DataFrame({"k": list(k_values), "inertia": [kmeans(k).fit(df).inertia_ for k in k_values]})


def silhouette(df, k_values=range(2, 11)):
    return pd.DataFrame({"k": list(k_values),
                         "silhouette": [silhouette_score(df, cluster(df, k)) for k in k_values]})
