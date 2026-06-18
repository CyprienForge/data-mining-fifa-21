import streamlit as st
import pandas as pd
from pathlib import Path

CACHE_FILE_CLEAN = Path("data/dataset_clean_with_streamlit.csv")
CACHE_FILE = Path("data/dataset_clean.csv")

def load_data(uploaded_file=None, is_clean=False):
    if CACHE_FILE_CLEAN.exists() and is_clean is False:
        return pd.read_csv(CACHE_FILE_CLEAN)
    
    if CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE)

    if uploaded_file is None:
        st.error("Veuillez importer le fichier OpenFoodFacts TSV.")
        st.stop()

    df = pd.read_csv(
        uploaded_file,
        sep="\t",
        on_bad_lines="skip"
    )

    df = clean(df)

    CACHE_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)
    return df

def clean(df):
    df = clean_columns(df)
    df = get_best_rows(df)
    return df

def get_best_rows(df, n=25000):
    df_scored = df.copy()
    df_scored["_score"] = df_scored.notna().sum(axis=1)
    df_sorted = df_scored.sort_values("_score", ascending=False).head(n)
    return df_sorted.drop(columns=["_score"])

def get_missing_values(df):
    missing = pd.DataFrame({
        "Colonne": df.columns,
        "Valeurs manquantes": df.isna().sum().values,
        "Pourcentage (%)": (df.isna().sum().values / len(df) * 100).round(2)
    })
    return missing.sort_values("Valeurs manquantes", ascending=False)

def clean_columns(df):
    seuil = 0.65
    df = df.dropna(thresh=len(df) * (1 - seuil), axis=1)

    return remove(df)

def remove(df):
    cols_a_supprimer = [
        "code",
        "url",
        "creator",
        "created_t",
        "created_datetime",
        "last_modified_t",
        "last_modified_datetime",

        "brands_tags",
        "countries_tags",
        "countries_en",
        "states_tags",
        "states_en",
        "additives_tags",
        "additives_en",

        "states",
        "ingredients_text",
        "serving_size"
    ]

    df = df.drop(columns=[c for c in cols_a_supprimer if c in df.columns])
    return df

def detect_type(serie: pd.DataFrame) -> str:
    s = pd.Series(serie).dropna()
    match s.dtype.name:
        case 'float64':
            if s.apply(float.is_integer).all():
                return "Quantitatif discret"
            return "Quantitatif continu"
        case 'int64':
            return "Quantitatif discret"
        case _:
            ORDINAL = ['faible', 'moyen', 'élevé', 'très élevé',
                       'petit', 'grand', 'jamais', 'parfois', 'souvent',
                       'toujours', 'mauvais', 'bien', 'très bien', 'excellent',
                       'débutant', 'intermédiaire', 'avancé', 'expert',
                       'xs', 's', 'm', 'l', 'xl', '2xl', '3xl']
            if any(v.lower() in ORDINAL for v in s):
                return 'Qualitatif ordinal'
            return 'Qualitatif nominal'
