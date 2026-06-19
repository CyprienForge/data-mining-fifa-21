import streamlit as st
import pandas as pd
from pathlib import Path

from pandas import Series, DataFrame
CACHE_FILE_CLEAN = Path("data/dataset_clean_with_streamlit.csv")
CACHE_FILE = Path("data/dataset_clean.csv")


def load_data(uploaded_file=None, is_clean=False):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = remove_lines(df)
        df = remove_columns(df)
        df = substitute_nan_by_zero(df)
        df.to_csv(CACHE_FILE, index=False)
        return df
    if CACHE_FILE_CLEAN.exists() and is_clean is False:
        return pd.read_csv(CACHE_FILE_CLEAN)
    
    if CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE)

    st.error("Veuillez importer le fichier players_21.csv.")
    st.stop()

def clean(df):
    df = clean_columns(df)
    df = get_best_rows(df)
    return df

def get_best_rows(df, n=100000):
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

    return remove_columns(df)

def remove_columns(df):
    cols_a_supprimer = [
        "dob",
        "nation_position",
        "team_position",
        "nation_jersey_number",
        "team_jersey_number",
        "sofifa_id",
        "player_url",
        "league_rank",
        "body_type",
        "real_face",
        "loaned_from",
        "nationality",
        "club_name",
        "league_name",
        "league_rank",
        "ls",
        "st",
        "rs",
        "lw",
        "lf",
        "cf",
        "rf",
        "rw",
        "lam",
        "cam",
        "ram",
        "lm",
        "lcm",
        "cm",
        "rcm",
        "rm",
        "lwb",
        "ldm",
        "cdm",
        "rdm",
        "rwb",
        "lb",
        "lcb",
        "cb",
        "rcb",
        "rb",
        "joined",
        "contract_valid_until",
        "gk_kicking",
        "gk_diving",
        "gk_speed",
        "gk_positioning",
        "gk_positionning",
        "gk_reflexes",
        "gk_handling",
        "goalkeeping_handling",
        "goalkeeping_kicking",
        "goalkeeping_positioning",
        "goalkeeping_reflexes",
        "player_tags",
        "player_traits",
        "nationality",
        "player_positions",
        "value_eur",
        "wage_eur",
        "release_clause_eur",
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
            if any(v.lower() in ORDINAL for v in s if isinstance(v, str)):
                return 'Qualitatif ordinal'
            return 'Qualitatif nominal'

def get_data_box(data: pd.DataFrame) :
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    borne_min = Q1 - 1.5 * IQR
    borne_max = Q3 + 1.5 * IQR
    outliers = data[(data < borne_min) | (data > borne_max)]
    return Q1, Q3, IQR, borne_min, borne_max, outliers

def substitute_nan_by_zero(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "gk_kicking",
        "gk_diving",
        "gk_speed",
        "gk_positioning",
        "gk_positionning",
        "gk_reflexes",
        "gk_handling",
        "physic",
        "passing",
        "defending",
        "shooting",
        "dribbling",
        "pace",
        "defending_marking",
        "loaned_from",
        "player_tags",
        "player_traits"
    ]
    df_clean = df.copy()

    columns_real = [col for col in columns if col in df_clean.columns]

    df_clean[columns_real] = df_clean[columns_real].fillna(0)

    return df_clean

def remove_lines(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["player_positions"].astype(str).str.contains("GK", na=False)].copy()
