import streamlit as st
from scipy.stats import skew, kurtosis
from functions import load_data, detect_type, get_data_box
import plotly.express as px

st.set_page_config(page_title="Visualisation", layout="wide")
st.title("Page 3 - Visualisation des données nettoyées")

df = load_data()

col_quant = [col for col in df.columns if detect_type(df[col]) in ["Quantitatif continu", "Quantitatif discret"]]
col_quali = [col for col in df.columns if detect_type(df[col]) in ["Qualitatif nominal", "Qualitatif ordinal"] and col != "product_name"]

option_plt = st.selectbox("Type de visualisation", ["Histogramme", "Boîte à moustaches", "Matrice de corrélation"], index=None, placeholder="Choisir la visualisation")

if option_plt is None:
    st.warning("Veuillez choisir un type de visualisation.")
    st.stop()

match option_plt:

    case "Histogramme":
        option_col = st.selectbox("Colonne", col_quant + col_quali, index=None, placeholder="Choisir la colonne")

        if option_col is None:
            st.warning("Veuillez sélectionner une colonne.")
            st.stop()

        st.subheader(f"Distribution de {option_col}")

        if option_col in col_quant:
            col1, col2 = st.columns(2)
            col1.metric("Asymétrie", round(skew(df[option_col].dropna()), 2))
            col2.metric("Aplatissement", round(kurtosis(df[option_col].dropna()), 2))
            fig = px.histogram(df, x=option_col, nbins=30, title=f"Distribution de {option_col}")
        else:
            fig = px.bar(df[option_col].value_counts().reset_index(), x=option_col, y="count", title=f"Distribution de {option_col}")

        st.plotly_chart(fig, use_container_width=True)

    case "Boîte à moustaches":
        option_col = st.selectbox("Colonne", col_quant, index=None, placeholder="Choisir la colonne")

        if option_col is None:
            st.warning("Veuillez sélectionner une colonne.")
            st.stop()

        data = df[option_col].dropna()
        Q1, Q3, IQR, borne_min, borne_max, outliers = get_data_box(data)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Min", f"{data.min():.2f}")
        col2.metric("Q1", f"{Q1:.2f}")
        col3.metric("Médiane", f"{data.median():.2f}")
        col4.metric("Q3", f"{Q3:.2f}")
        col5.metric("Max", f"{data.max():.2f}")

        fig = px.box(df, y=option_col, points="outliers")
        st.plotly_chart(fig, use_container_width=True)

        if not outliers.empty:
            with st.expander(f"{len(outliers)} valeurs aberrantes [{borne_min:.2f} ; {borne_max:.2f}]"):
                st.dataframe(outliers.reset_index(), use_container_width=True, hide_index=True)

    case "Matrice de corrélation":
        corr = df[col_quant].dropna().corr()

        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                        height=700)
        st.plotly_chart(fig, use_container_width=True)

        threshold = st.slider("Seuil |k| minimum", 0.0, 1.0, 0.7, 0.05)

        strong = corr.unstack().reset_index()
        strong.columns = ["Var A", "Var B", "k"]
        strong = strong[(strong["Var A"] < strong["Var B"]) & (strong["k"].abs() >= threshold)]
        strong = strong.sort_values("k", key=abs, ascending=False)

        st.dataframe(strong, use_container_width=True, hide_index=True)