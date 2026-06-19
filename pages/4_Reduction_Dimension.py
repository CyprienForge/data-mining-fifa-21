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
import sys

st.set_page_config(
    page_title="Réduction de dimension",
    layout="wide",
)

st.title("Page 4 - Réduction de dimension")

df = load_data()
print(df)
st.header("Analyse en Composantes Principales (ACP)")

# Sélection des colonnes numériques
df_num = df.select_dtypes(include=np.number)

if len(df_num.columns) < 2:
    st.error("Il faut au moins 2 variables numériques pour réaliser une ACP.")
    st.stop()

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

# ACP
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

st.subheader("Variance expliquée")
variance_expliquee = pca.explained_variance_ratio_ * 100
variance_df = pd.DataFrame({
    "Composante": [f"CP{i+1}" for i in range(len(variance_expliquee))],
    "Variance expliquée (%)": variance_expliquee
})
st.dataframe(variance_df)

st.subheader("Diagramme des éboulis")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    range(1, len(variance_expliquee) + 1),
    variance_expliquee,
    marker="o"
)
ax.set_xlabel("Composantes principales")
ax.set_ylabel("Variance expliquée (%)")
ax.set_title("Diagramme des éboulis")
ax.grid(True)
st.pyplot(fig)

st.subheader("Cercle des corrélations")
fig, ax = plt.subplots(figsize=(8, 8))
correlations = pca.components_.T[:, :2]
circle = plt.Circle((0, 0), 1, fill=False)
ax.add_artist(circle)
for i, variable in enumerate(df_num.columns):
    ax.arrow(
        0,
        0,
        correlations[i, 0],
        correlations[i, 1],
        head_width=0.03,
        length_includes_head=True
    )
    ax.text(
        correlations[i, 0] * 1.1,
        correlations[i, 1] * 1.1,
        variable,
        fontsize=8
    )
ax.axhline(0)
ax.axvline(0)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel(
    f"F1 ({variance_expliquee[0]:.2f}%)"
)
ax.set_ylabel(
    f"F2 ({variance_expliquee[1]:.2f}%)"
)
ax.set_title("Cercle des corrélations")
st.pyplot(fig)

st.subheader("Projection des individus sur le plan factoriel (F1, F2)")
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, color="gray")

distances = np.sqrt(X_pca[:, 0] ** 2 + X_pca[:, 1] ** 2)

seuil_distance = np.percentile(distances, 99.5)

noms_joueurs = df["long_name"].values

for i, nom in enumerate(noms_joueurs):
    if distances[i] >= seuil_distance:
        ax.annotate(
            nom,
            (X_pca[i, 0], X_pca[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            alpha=0.9,
        )
        ax.scatter(X_pca[i, 0], X_pca[i, 1], color="red", s=30)

ax.set_xlabel(f"F1 ({variance_expliquee[0]:.2f}%)")
ax.set_ylabel(f"F2 ({variance_expliquee[1]:.2f}%)")
ax.set_title("Projection des individus (Zoom sur les profils atypiques)")
ax.grid(True)
st.pyplot(fig)

# ISOMAP (Méthode non-linéaire)
st.header("Isomap (Réduction non-linéaire)")

st.write(
    "Isomap cherche à préserver les distances géodésiques (le long de la structure des données) "
    "plutôt que les distances euclidiennes directes."
)

n_samples = X_scaled.shape[0]

MAX_SAMPLES_ISOMAP = 3000 

if n_samples > MAX_SAMPLES_ISOMAP:
    st.warning(
        f"⚠️ Le jeu de données contient {n_samples} lignes. Isomap ayant une complexité algorithmique élevée, "
        f"le calcul a été limité à un échantillon aléatoire de {MAX_SAMPLES_ISOMAP} individus pour éviter les blocages."
    )
    np.random.seed(42)
    indices_subsample = np.random.choice(n_samples, MAX_SAMPLES_ISOMAP, replace=False)
    X_scaled_iso = X_scaled[indices_subsample]
    c_vector = X_pca[indices_subsample, 0]
else:
    X_scaled_iso = X_scaled
    c_vector = X_pca[:, 0]

n_samples_iso = X_scaled_iso.shape[0]
max_neighbors = min(100, n_samples_iso - 1)

if max_neighbors > 2:
    n_neighbors = st.slider(
        "Nombre de voisins (k)", 
        min_value=2, 
        max_value=max_neighbors, 
        value=min(15, max_neighbors),
        step=1,
        help="Un k trop petit peut diviser le graphe en plusieurs morceaux isolés."
    )
else:
    n_neighbors = 2

connectivity = kneighbors_graph(X_scaled_iso, n_neighbors=n_neighbors, mode='connectivity')
n_components, labels = connected_components(csgraph=connectivity, directed=False)

@st.cache_data(show_spinner=False)
def run_isomap(data, k):
    iso = Isomap(n_neighbors=k, n_components=2, path_method='auto', neighbors_algorithm='auto', n_jobs=-1)
    X_iso = iso.fit_transform(data)
    return X_iso, iso.reconstruction_error()

with st.spinner("Calcul de l'Isomap en cours (Optimisé)..."):
    try:
        X_isomap, rec_error = run_isomap(X_scaled_iso, n_neighbors)
        
        st.subheader(f"Projection Isomap (k = {n_neighbors})")
        
        fig_iso, ax_iso = plt.subplots(figsize=(10, 6))
        ax_iso.scatter(
            X_isomap[:, 0],
            X_isomap[:, 1],
            alpha=0.6,
            c=c_vector,
            cmap='viridis'
        )
        ax_iso.set_xlabel("Isomap Dimension 1")
        ax_iso.set_ylabel("Isomap Dimension 2")
        ax_iso.set_title(f"Projection via Isomap (k={n_neighbors})")
        ax_iso.grid(True)
        
        cbar = fig_iso.colorbar(ax_iso.collections[0], ax=ax_iso)
        cbar.set_label('Gradient basé sur la CP1 de l’ACP')
        
        st.pyplot(fig_iso)

    except Exception as e:
        st.error(f"Une erreur est survenue lors du calcul d'Isomap : {e}")

st.header("Projection ACP à 5 composantes (visualisée en 2D)")

pca_5 = PCA(n_components=5)
X_pca5 = pca_5.fit_transform(X_scaled)

variance_5 = pca_5.explained_variance_ratio_ * 100
cumvar_5 = np.cumsum(variance_5)

st.markdown(f"""
- **Variance cumulée à 5 composantes** : `{cumvar_5[-1]:.1f}%`
- CP1 : `{variance_5[0]:.1f}%` | CP2 : `{variance_5[1]:.1f}%` | CP3 : `{variance_5[2]:.1f}%` | CP4 : `{variance_5[3]:.1f}%` | CP5 : `{variance_5[4]:.1f}%`
""")

fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.scatter(X_pca5[:, 0], X_pca5[:, 1], alpha=0.3, s=5, color="steelblue")

distances_5 = np.sqrt(X_pca5[:, 0] ** 2 + X_pca5[:, 1] ** 2)
seuil_5 = np.percentile(distances_5, 99.5)
noms = df["long_name"].values

for i, nom in enumerate(noms):
    if distances_5[i] >= seuil_5:
        ax5.annotate(nom, (X_pca5[i, 0], X_pca5[i, 1]),
                     xytext=(5, 5), textcoords="offset points",
                     fontsize=9, fontweight="bold", alpha=0.9)
        ax5.scatter(X_pca5[i, 0], X_pca5[i, 1], color="red", s=30)

ax5.set_xlabel(f"CP1 ({variance_5[0]:.1f}%)")
ax5.set_ylabel(f"CP2 ({variance_5[1]:.1f}%)")
ax5.set_title("Projection des individus - ACP 5 composantes (plan CP1/CP2)")
ax5.grid(True)
st.pyplot(fig5)