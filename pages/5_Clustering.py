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
    n_components_pca = st.slider(
        "Nombre de composantes ACP", 
        min_value=2, 
        max_value=min(10, X_scaled.shape[1]), 
        value=5
    )
    pca = PCA(n_components=n_components_pca)
    X_work = pca.fit_transform(X_scaled)
    st.caption(
        f"Variance expliquée cumulée : {pca.explained_variance_ratio_.sum():.1%}"
    )
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

st.header("Indices d'évaluation")

if n_clusters >= 2:
    mask_valid = labels != -1
    X_eval, labels_eval = X_work[mask_valid], labels[mask_valid]
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette ↑", f"{silhouette_score(X_eval, labels_eval):.3f}")
    c2.metric("Davies-Bouldin ↓", f"{davies_bouldin_score(X_eval, labels_eval):.3f}")
    c3.metric("Calinski-Harabasz ↑", f"{calinski_harabasz_score(X_eval, labels_eval):.0f}")

if method in ("K-Means", "Spectral Clustering"):
    k_range = range(2, 11)
    sils, dbs, chs, inertias = [], [], [], []

    for ki in k_range:
        if method == "K-Means":
            m = KMeans(ki, n_init=10, random_state=42)
        else:
            m = SpectralClustering(n_clusters=ki, affinity=affinity, gamma=gamma, n_neighbors=n_neighbors_spec, random_state=42, assign_labels="kmeans")

        lbl = m.fit_predict(X_work)
        sils.append(silhouette_score(X_work, lbl))
        dbs.append(davies_bouldin_score(X_work, lbl))
        chs.append(calinski_harabasz_score(X_work, lbl))

        if method == "K-Means":
            inertias.append(m.inertia_)

    if method == "K-Means":
        n_plots = 4
    else :
        nb_plots = 3

    fig2, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))

    for ax_, data, title in zip(axes, [sils, dbs, chs], ["Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑"]):
        ax_.plot(list(k_range), data, marker="o")
        ax_.set_title(title); ax_.set_xlabel("k")

    if method == "K-Means":
        axes[-1].plot(list(k_range), inertias, marker="o")
        axes[-1].set_title("Inertie (coude)"); axes[-1].set_xlabel("k")

    plt.tight_layout()
    st.pyplot(fig2)

st.header("Statistiques des clusters")

df_work = pd.DataFrame(X_raw.values, columns=numeric_cols)
df_work["_cluster"] = labels

for lbl in unique_labels:
    sub = df_work[df_work["_cluster"] == lbl][numeric_cols]
    st.subheader(f"Cluster {lbl} — {len(sub)} points")

    moyennes = sub.mean().rename("Moyenne")
    ecarts = sub.std().rename("Écart-type")
    stats = pd.concat([moyennes, ecarts], axis=1).T

    if method == "K-Means":
        stats.loc["Centroïde"] = model.cluster_centers_[lbl]

    st.dataframe(stats, use_container_width=True)

if -1 in labels:
    st.write(f"**Points bruit :** {(labels == -1).sum()}")

st.header("Interprétation métier")

global_mean = df_work[numeric_cols].mean()
global_std = df_work[numeric_cols].std().replace(0, 1)

for lbl in unique_labels:
    sub = df_work[df_work["_cluster"] == lbl][numeric_cols]
    z = (sub.mean() - global_mean) / global_std
    top_high = z.nlargest(3).index.tolist()
    top_low = z.nsmallest(3).index.tolist()
    pct = len(sub) / len(df_work) * 100
    desc = (f"**Cluster {lbl}** ({len(sub)} points, {pct:.1f}% du jeu) : "
            f"valeurs élevées pour {', '.join(f'*{c}*' for c in top_high)} ; "
            f"valeurs faibles pour {', '.join(f'*{c}*' for c in top_low)}.")
    st.markdown(desc)