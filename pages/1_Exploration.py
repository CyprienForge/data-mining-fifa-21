import pandas as pd
import streamlit as st
from functions import load_data, detect_type
from pathlib import Path

CACHE_FILE = Path("data/dataset_clean.csv")

st.title("Partie I : Exploration initiale des données")



df = None

if not CACHE_FILE.exists():

    st.header("Chargement des données")

    fichier = st.file_uploader(
        "Importer le CSV players_21.csv",
        type=["tsv", "csv"]
    )

    if fichier:
        df = load_data(fichier)
else:
    df = pd.read_csv(CACHE_FILE)

if df is not None:
    st.header("Aperçu des données")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Premières lignes")
        st.dataframe(df.head(10))
    with col2:
        st.subheader("Dernières lignes")
        st.dataframe(df.tail(10))

    st.header("Résumé statistique")

    c1, c2 = st.columns(2)
    c1.metric("Nombre de lignes", df.shape[0])
    c2.metric("Nombre de colonnes", df.shape[1])

    st.subheader("Description des colonnes")

    resume = pd.DataFrame({
        "Colonne": df.columns,
        "Type pandas": df.dtypes.values.astype(str),
        "Type statistique": [detect_type(df[col]) for col in df.columns],
        "Valeurs manquantes": df.isna().sum().values,
        "% manquant": (df.isna().sum().values / len(df) * 100).round(2),
        "Valeurs uniques": df.nunique().values
    })

    st.dataframe(resume, use_container_width=True)

    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe().T.round(2), use_container_width=True)