import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from functions import load_data

st.set_page_config(page_title="Clustering", layout="wide")
st.title("Page 5 - Clustering")

df = load_data()
if df is None or df.empty:
    st.warning("Aucune donnée chargée.")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
X_raw = df[numeric_cols].dropna()
X_scaled = StandardScaler().fit_transform(X_raw)

st.header("Configuration")
col1, col2 = st.columns(2)
with col1:
    workspace = st.radio("Espace de travail", ["Données brutes standardisées", "Données projetées par ACP"])
with col2:
    method = st.selectbox("Algorithme", ["K-Means", "DBSCAN", "Spectral Clustering"])

if workspace == "Données projetées par ACP":
    n_pca = st.slider("Composantes ACP", 2, min(10, X_scaled.shape[1]), 2)
    pca_work = PCA(n_components=n_pca)
    X_work = pca_work.fit_transform(X_scaled)
    st.caption(f"Variance expliquée cumulée : {pca_work.explained_variance_ratio_.sum():.1%}")
else:
    X_work = X_scaled

st.header("Hyperparamètres")

affinity, gamma, n_neighbors_spec = "rbf", 1.0, 10

if method == "K-Means":
    k = st.slider("k", 2, 15, 3)
    init_method = st.selectbox("Initialisation", ["k-means++", "random"])
    max_iter = st.slider("Itérations max", 100, 1000, 300, 100)
    n_init = st.slider("n_init", 1, 20, 10)
    model = KMeans(k, init=init_method, max_iter=max_iter, n_init=n_init, random_state=42)

elif method == "DBSCAN":
    eps = st.slider("Epsilon (ε)", 0.1, 5.0, 0.5, 0.1)
    min_samples = st.slider("MinPts", 2, 20, 5)
    metric = st.selectbox("Métrique", ["euclidean", "manhattan", "cosine"])
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

else:
    k = st.slider("k", 2, 15, 3)
    affinity = st.selectbox("Affinité", ["rbf", "nearest_neighbors", "cosine"])
    gamma = st.slider("Gamma", 0.01, 5.0, 1.0, 0.01) if affinity == "rbf" else 1.0
    n_neighbors_spec = st.slider("n_neighbors", 2, 20, 10) if affinity == "nearest_neighbors" else 10
    model = SpectralClustering(n_clusters=k, affinity=affinity, gamma=gamma,
                               n_neighbors=n_neighbors_spec, random_state=42, assign_labels="kmeans")

labels = np.array(model.fit_predict(X_work))
unique_labels = sorted([l for l in np.unique(labels) if l != -1])
n_clusters = len(unique_labels)

if method == "DBSCAN":
    st.metric("Clusters détectés", n_clusters)
    st.metric("Points bruit", (labels == -1).sum())

st.header("Visualisation des clusters")

dim = st.radio("Dimension", ["2D", "3D"], horizontal=True)
n_dim = 3 if dim == "3D" else 2
pca_viz = PCA(n_components=n_dim)
X_viz = pca_viz.fit_transform(X_work)
cmap = plt.colormaps["tab10"].resampled(max(n_clusters, 1))

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection="3d" if dim == "3D" else None)

for i, lbl in enumerate(unique_labels):
    m = labels == lbl
    coords = (X_viz[m, 0], X_viz[m, 1]) if dim == "2D" else (X_viz[m, 0], X_viz[m, 1], X_viz[m, 2])
    ax.scatter(*coords, s=15, color=cmap(i), label=f"Cluster {lbl}", alpha=0.7)

if method == "K-Means":
    c_viz = pca_viz.transform(model.cluster_centers_)
    coords_c = (c_viz[:, 0], c_viz[:, 1]) if dim == "2D" else (c_viz[:, 0], c_viz[:, 1], c_viz[:, 2])
    ax.scatter(*coords_c, s=200, marker="X", color="black", zorder=5, label="Centroïdes")

if -1 in labels:
    m = labels == -1
    coords = (X_viz[m, 0], X_viz[m, 1]) if dim == "2D" else (X_viz[m, 0], X_viz[m, 1], X_viz[m, 2])
    ax.scatter(*coords, s=10, color="grey", alpha=0.3, label="Bruit")

ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
if dim == "3D": ax.set_zlabel("PC3")
ax.legend(fontsize=7)
st.pyplot(fig)
