import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from functions import load_data
from sklearn.manifold import MDS
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(
    page_title="Clustering",
    layout="wide",
)
st.title("Page 5 - Clustering")

method = st.selectbox(
    "Algorithme de clustering",
    [
        "K-Means",
        "DBSCAN",
        "Spectral Clustering"
    ],
)

df = load_data()
if df is None or df.empty:
    st.warning("Aucune donnée chargée.")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
X_raw = df[numeric_cols].dropna()

st.header("Configuration")

workspace = st.radio(
    "Espace de travail",
    ["Données brutes standardisées", "Données projetées par ACP"],
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

if workspace == "Données projetées par ACP":
    n_components_pca = st.slider(
        "Nombre de composantes ACP", min_value=2, max_value=min(10, X_scaled.shape[1]), value=2
    )
    pca = PCA(n_components=n_components_pca)
    X_work = pca.fit_transform(X_scaled)
    st.caption(
        f"Variance expliquée cumulée : {pca.explained_variance_ratio_.sum():.1%}"
    )
else:
    X_work = X_scaled

st.header("🔧 Hyperparamètres")

labels = None

if method == "K-Means":
    k = st.slider("Nombre de clusters (k)", min_value=2, max_value=15, value=3)
    init_method = st.selectbox("Initialisation", ["k-means++", "random"])
    max_iter = st.slider("Itérations max", min_value=100, max_value=1000, value=300, step=100)
    n_init = st.slider("Nombre d'initialisations (n_init)", min_value=1, max_value=20, value=10)

    model = KMeans(k, init=init_method, max_iter=max_iter, n_init=n_init, random_state=42)
    labels = model.fit_predict(X_work)

elif method == "DBSCAN":
    eps = st.slider("Epsilon (ε)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    min_samples = st.slider("MinPts", min_value=2, max_value=20, value=5)
    metric = st.selectbox("Métrique de distance", ["euclidean", "manhattan", "cosine"])

    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X_work)

    n_noise = (labels == -1).sum()
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    st.metric("Clusters détectés", n_clusters_found)
    st.metric("Points bruit (label -1)", n_noise)

elif method == "Spectral Clustering":
    k = st.slider("Nombre de clusters (k)", min_value=2, max_value=15, value=3)
    affinity = st.selectbox(
        "Affinité", ["rbf", "nearest_neighbors", "cosine"]
    )
    if affinity == "rbf":
        gamma = st.slider("Gamma", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
    else:
        gamma = 1.0
    if affinity == "nearest_neighbors":
        n_neighbors_spec = st.slider("n_neighbors", min_value=2, max_value=20, value=10)
    else:
        n_neighbors_spec = 10

    model = SpectralClustering(
        n_clusters=k,
        affinity=affinity,
        gamma=gamma,
        n_neighbors=n_neighbors_spec,
        random_state=42,
        assign_labels="kmeans",
    )
    labels = model.fit_predict(X_work)
