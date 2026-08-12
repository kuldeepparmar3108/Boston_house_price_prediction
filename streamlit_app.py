import streamlit as st
import pandas as pd
import joblib


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# -----------------------------------
# Load Trained Model
# -----------------------------------

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()


# -----------------------------------
# App Title
# -----------------------------------

st.title("🏠 Boston House Price Predictor")

st.write(
    "Enter the details below to predict the value "
    "of a house."
)


# -----------------------------------
# Input Form
# -----------------------------------

with st.form("house_prediction_form"):

    st.subheader("House Details")

    crim = st.number_input(
        "CRIM - Crime rate",
        min_value=0.0,
        value=3.61,
        step=0.01
    )

    zn = st.number_input(
        "ZN - Residential land zoned (%)",
        min_value=0.0,
        value=11.36,
        step=0.01
    )

    indus = st.number_input(
        "INDUS - Non-retail business acres (%)",
        min_value=0.0,
        value=11.14,
        step=0.01
    )

    chas = st.selectbox(
        "CHAS - Bounds Charles River?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    nox = st.number_input(
        "NOX - Nitric oxide concentration",
        min_value=0.0,
        value=0.55,
        step=0.01
    )

    rm = st.number_input(
        "RM - Average number of rooms",
        min_value=0.0,
        value=6.28,
        step=0.01
    )

    age = st.number_input(
        "AGE - Older homes (%)",
        min_value=0.0,
        max_value=100.0,
        value=68.0,
        step=0.1
    )

    dis = st.number_input(
        "DIS - Distance to employment centers",
        min_value=0.0,
        value=3.80,
        step=0.01
    )

    rad = st.number_input(
        "RAD - Highway accessibility index",
        min_value=0.0,
        value=4.0,
        step=1.0
    )

    tax = st.number_input(
        "TAX - Property tax rate",
        min_value=0.0,
        value=300.0,
        step=1.0
    )

    ptratio = st.number_input(
        "PTRATIO - Pupil-teacher ratio",
        min_value=0.0,
        value=18.7,
        step=0.1
    )

    black = st.number_input(
        "BLACK - Demographic index",
        min_value=0.0,
        value=390.0,
        step=0.1
    )

    lstat = st.number_input(
        "LSTAT - Lower-status population (%)",
        min_value=0.0,
        max_value=100.0,
        value=12.0,
        step=0.1
    )

    predict_button = st.form_submit_button(
        "🏠 Predict House Value",
        use_container_width=True
    )


# -----------------------------------
# Prediction
# -----------------------------------

if predict_button:

    input_data = pd.DataFrame([[
        crim,
        zn,
        indus,
        chas,
        nox,
        rm,
        age,
        dis,
        rad,
        tax,
        ptratio,
        black,
        lstat
    ]])

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")

    st.metric(
        label="Estimated House Value",
        value=f"${prediction:.2f}k"
    )

    st.info(
        "The predicted value is in thousands of US dollars "
        "(e.g., $25.50k = $25,500)."
    )