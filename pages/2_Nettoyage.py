import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
from functions import load_data
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler

st.set_page_config(
    page_title="Nettoyage",
    layout="wide",
)
st.title("Page 2 - Nettoyage des données")

df = load_data(uploaded_file=None, is_clean=True)
df_clean = df.copy()

st.subheader("Aperçu des données")
st.dataframe(df)

st.divider()
st.subheader("1. Diagnostic des valeurs manquantes")

n_missing = df.isnull().sum()
pct_missing = (n_missing / len(df) * 100).round(2)
recap = pd.DataFrame({
    "Type": df.dtypes.astype(str),
    "Valeurs manquantes": n_missing,
    "Pourcentage (%)": pct_missing,
}).sort_values("Valeurs manquantes", ascending=False)

st.dataframe(recap)

cols_with_na = recap[recap["Valeurs manquantes"] > 0].index.tolist()

if not cols_with_na:
    st.success("Aucune valeur manquante détectée : aucun nettoyage n'est nécessaire.")
else:
    st.warning(
        f"{len(cols_with_na)} colonne(s) contiennent des valeurs manquantes : "
        f"{', '.join(cols_with_na)}"
    )

    st.divider()
    st.subheader("2. Choix de la méthode de traitement")

    method = st.selectbox(
        "Méthode à appliquer",
        [
            "Remplacer par le mode",
            "Remplacer par la moyenne",
            "Remplacer par la médiane",
            "Imputation KNN (k plus proches voisins)",
            "Imputation par régression linéaire",
            "Supprimer les lignes contenant des valeurs manquantes",
            "Supprimer les colonnes contenant des valeurs manquantes"
        ],
    )

    df_clean = df.copy()
    numeric_cols_with_na = [
        c for c in cols_with_na if c in df.select_dtypes(include=np.number).columns
    ]

    # --- Suppression des lignes ---
    if method == "Supprimer les lignes contenant des valeurs manquantes":
        subset = st.multiselect(
            "Limiter la suppression aux colonnes suivantes (vide = toutes les colonnes)",
            options=cols_with_na,
        )
        df_clean = df_clean.dropna(subset=subset if subset else None)

    # --- Suppression des colonnes ---
    elif method == "Supprimer les colonnes contenant des valeurs manquantes":
        seuil = st.slider(
            "Seuil de tolérance : supprimer les colonnes ayant plus de X% de valeurs manquantes",
            min_value=0, max_value=100, value=0, step=5,
        )
        limite = int(len(df_clean) * (1 - seuil / 100))
        df_clean = df_clean.dropna(axis=1, thresh=limite)

    # --- Moyenne ---
    elif method == "Remplacer par la moyenne":
        cols = st.multiselect(
            "Colonnes numériques à imputer",
            options=numeric_cols_with_na,
            default=numeric_cols_with_na,
        )
        for col in cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    # --- Médiane ---
    elif method == "Remplacer par la médiane":
        cols = st.multiselect(
            "Colonnes numériques à imputer",
            options=numeric_cols_with_na,
            default=numeric_cols_with_na,
        )
        for col in cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # --- Mode ---
    elif method == "Remplacer par le mode":
        cols = st.multiselect(
            "Colonnes à imputer",
            options=cols_with_na,
            default=cols_with_na,
        )
        for col in cols:
            mode_vals = df_clean[col].mode()
            if not mode_vals.empty:
                df_clean[col] = df_clean[col].fillna(mode_vals.iloc[0])

    # --- KNN ---
    elif method == "Imputation KNN (k plus proches voisins)":
        if not numeric_cols_with_na:
            st.error(
                "Aucune colonne numérique avec valeurs manquantes : "
                "l'imputation KNN ne s'applique qu'aux colonnes numériques."
            )
        else:
            k = st.slider("Nombre de voisins (k)", min_value=1, max_value=15, value=5)
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            imputer = KNNImputer(n_neighbors=k)
            df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])

    # --- Régression linéaire ---
    elif method == "Imputation par régression linéaire":
        if not numeric_cols_with_na:
            st.error(
                "Aucune colonne numérique avec valeurs manquantes : "
                "cette méthode ne s'applique qu'aux colonnes numériques."
            )
        else:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            imputer = IterativeImputer(
                estimator=LinearRegression(), random_state=0, max_iter=10
            )
            df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])

    st.divider()
    st.subheader("3. Résultat après nettoyage")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes avant", len(df))
    c2.metric("Lignes après", len(df_clean))
    c3.metric("Val. manquantes avant", int(df.isnull().sum().sum()))
    c4.metric("Val. manquantes après", int(df_clean.isnull().sum().sum()))

    st.dataframe(df_clean)

# 2. DÉTECTION ET TRAITEMENT DES OUTLIERS
st.divider()
st.header("2. Détection et traitement des outliers")


def detect_iqr(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper), lower, upper


def detect_zscore(series, threshold=3.0):
    mean, std = series.mean(), series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(False, index=series.index), mean, mean
    z = (series - mean) / std
    lower, upper = mean - threshold * std, mean + threshold * std
    return z.abs() > threshold, lower, upper

numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols:
    st.info("Aucune colonne numérique disponible pour la détection d'outliers.")
else:
    detection_method = st.radio(
        "Méthode de détection",
        ["Z-score", "IQR (écart interquartile)"],
        horizontal=True,
        key="outlier_detection_method",
    )

    if detection_method == "IQR (écart interquartile)":
        threshold = st.slider("Multiplicateur de l'IQR (k)", 1.0, 3.0, 1.5, 0.1, key="iqr_k")
        detect_func = lambda s: detect_iqr(s, threshold)
    else:
        threshold = st.slider("Seuil |z|", 1.0, 5.0, 3.0, 0.1, key="z_threshold")
        detect_func = lambda s: detect_zscore(s, threshold)

    cols_to_check = st.multiselect(
        "Colonnes numériques à analyser",
        options=numeric_cols,
        default=numeric_cols,
        key="outlier_cols",
    )

    if cols_to_check:
        masks, bounds = {}, {}
        for col in cols_to_check:
            mask, lower, upper = detect_func(df_clean[col])
            masks[col] = mask
            bounds[col] = (lower, upper)

        summary = pd.DataFrame({
            "Colonne": cols_to_check,
            "Nb outliers": [int(masks[c].sum()) for c in cols_to_check],
            "% outliers": [round(masks[c].mean() * 100, 2) for c in cols_to_check],
            "Borne inf.": [round(bounds[c][0], 3) for c in cols_to_check],
            "Borne sup.": [round(bounds[c][1], 3) for c in cols_to_check],
        })
        st.dataframe(summary)

        treatment = st.radio(
            "Stratégie de traitement",
            [
                "Ne rien faire (visualisation uniquement)",
                "Supprimer les lignes contenant des outliers",
                "Remplacer par la médiane de la colonne",
            ],
            key="outlier_treatment",
        )

        nb_lines = len(df_clean)
        if treatment == "Supprimer les lignes contenant des outliers":
            combined_mask = pd.Series(False, index=df_clean.index)
            for col in cols_to_check:
                combined_mask |= masks[col]
            df_clean = df_clean[~combined_mask]
        elif treatment == "Remplacer par la médiane de la colonne":
            for col in cols_to_check:
                median_val = df_clean[col].median()
                df_clean.loc[masks[col], col] = median_val

        st.write(f"**Lignes avant traitement :** {nb_lines} → **Lignes après :** {len(df_clean)}")
        st.dataframe(df_clean)

# 3. ENCODAGE DES VARIABLES QUALITATIVES
st.divider()
st.header("3. Encodage des variables qualitatives")
 
categorical_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
 
if not categorical_cols:
    st.info("Aucune variable qualitative détectée : encodage non nécessaire.")
else:
    cardinality = df_clean[categorical_cols].nunique().sort_values(ascending=False)
    st.dataframe(pd.DataFrame({"Nb de valeurs uniques": cardinality}))
 
    with st.expander("Quelle méthode choisir ? (discussion)"):
        st.markdown(
            "Étant donné que les machines ne comprennent que des nombres, nous devons évidemment encoder les variables qualitatives pour" \
            "les rendres manipulables. Étant donné que nos colonnes ne sont pas ordinales (on ne peut pas les classer), on privilégiera le **One-Hot Encoding**"
        )
 
    cols_to_encode = st.multiselect(
        "Colonnes à encoder",
        options=categorical_cols,
        default=categorical_cols,
        key="encode_cols",
    )
 
    encoding_method = st.radio(
        "Méthode d'encodage",
        ["Aucun (laisser tel quel)", "One-Hot Encoding"],
        horizontal=True,
        key="encoding_method",
    )
 
    if cols_to_encode and encoding_method != "Aucun (laisser tel quel)":
        remaining_na = df_clean[cols_to_encode].isnull().sum().sum()
        if remaining_na > 0:
            st.info(
                f"{remaining_na} valeur(s) manquante(s) restent dans les colonnes sélectionnées ; "
                "pensez à les traiter à l'étape 1 si besoin avant l'encodage."
            )
 
        drop_first = st.checkbox(
            "Supprimer une catégorie de référence par colonne (drop_first, évite la colinéarité)",
            value=False,
            key="drop_first",
        )
        df_clean = pd.get_dummies(
            df_clean, columns=cols_to_encode, drop_first=drop_first, dtype=int
        )
 
    st.dataframe(df_clean)

# 4. NORMALISATION DES DONNÉES
st.divider()
st.header("4. Normalisation des données")
 
numeric_cols_norm = df_clean.select_dtypes(include=np.number).columns.tolist()
 
if not numeric_cols_norm:
    st.info("Aucune colonne numérique disponible pour la normalisation.")
else:
    norm_method = st.selectbox(
        "Méthode de normalisation",
        [
            "Aucune (laisser les données telles quelles)",
            "Min-Max (rescale entre 0 et 1)",
            "Standardisation Z-score (centrer-réduire)",
            "RobustScaler (basé sur la médiane et l'IQR)",
        ],
        key="norm_method",
    )
 
    if norm_method == "Aucune (laisser les données telles quelles)":
        st.dataframe(df_clean)
    else:
        default_cols = [c for c in numeric_cols_norm if df_clean[c].nunique() > 2]
        cols_to_normalize = st.multiselect(
            "Colonnes à normaliser (les colonnes binaires issues du one-hot encoding sont "
            "décochées par défaut, car les normaliser n'a généralement pas de sens)",
            options=numeric_cols_norm,
            default=default_cols if default_cols else numeric_cols_norm,
            key="norm_cols",
        )
 
        if cols_to_normalize:
            stats_avant = df_clean[cols_to_normalize].agg(["min", "max", "mean", "std"]).T.round(3)
            st.write("**Statistiques avant normalisation :**")
            st.dataframe(stats_avant)
 
            if norm_method == "Min-Max (rescale entre 0 et 1)":
                scaler = MinMaxScaler()
            elif norm_method == "Standardisation Z-score (centrer-réduire)":
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
 
            df_clean[cols_to_normalize] = scaler.fit_transform(df_clean[cols_to_normalize])
 
            stats_apres = df_clean[cols_to_normalize].agg(["min", "max", "mean", "std"]).T.round(3)
            st.write("**Statistiques après normalisation :**")
            st.dataframe(stats_apres)
 
            col_to_plot_norm = st.selectbox(
                "Visualiser la distribution d'une colonne après normalisation",
                cols_to_normalize,
                key="norm_plot_col",
            )
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.hist(df_clean[col_to_plot_norm].dropna(), bins=30)
            ax.set_title(f"Distribution après normalisation — {col_to_plot_norm}")
            st.pyplot(fig)
 
            st.dataframe(df_clean)
        else:
            st.dataframe(df_clean)


# Conclusion de la page
st.divider()
st.header("Conclusion du nettoyage")
 
c1, c2 = st.columns(2)
c1.metric("Lignes (départ → final)", f"{len(df)} → {len(df_clean)}")
c2.metric("Colonnes (départ → final)", f"{df.shape[1]} → {df_clean.shape[1]}")
 
if st.button("Valider ces données nettoyées pour les pages suivantes"):
    st.session_state["df_clean"] = df_clean
    df_clean.to_csv("data/dataset_clean_with_streamlit.csv", index=False, encoding="utf-8")
    st.success("Le fichier CSV a été créé dans le dossier data/.")