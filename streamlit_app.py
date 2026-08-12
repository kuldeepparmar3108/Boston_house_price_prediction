import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor



# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏘️ Housing Price Prediction")
st.write("Predictive analysis for the price of a house")


# -----------------------------
# Load dataset
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()


""" # Remove unwanted index column if present
df = df.loc[:, ~df.columns.str.contains("^Unnamed")] """


""" # -----------------------------
# Prepare model
# -----------------------------

target = "medv"

X = df.drop(columns=[target])
y = df[target]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y) """


# -----------------------------
# App title
# -----------------------------

st.title("🏠 Boston House Price Predictor")

st.write(
    "Enter the property and neighborhood details below "
    "to estimate the median house value."
)


# -----------------------------
# Input form
# -----------------------------

st.subheader("Enter House Details")

with st.form("prediction_form"):

    crim = st.number_input(
        "CRIM — Crime rate",
        min_value=0.0,
        value=float(df["crim"].median()),
        step=0.01
    )

    zn = st.number_input(
        "ZN — Residential land zoned (%)",
        min_value=0.0,
        value=float(df["zn"].median()),
        step=0.1
    )

    indus = st.number_input(
        "INDUS — Non-retail business acres (%)",
        min_value=0.0,
        value=float(df["indus"].median()),
        step=0.1
    )

    chas = st.selectbox(
        "CHAS — Bounds Charles River?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    nox = st.number_input(
        "NOX — Nitric oxide concentration",
        min_value=0.0,
        value=float(df["nox"].median()),
        step=0.01
    )

    rm = st.number_input(
        "RM — Average number of rooms",
        min_value=1.0,
        value=float(df["rm"].median()),
        step=0.1
    )

    age = st.number_input(
        "AGE — Older homes (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(df["age"].median()),
        step=1.0
    )

    dis = st.number_input(
        "DIS — Distance to employment centers",
        min_value=0.0,
        value=float(df["dis"].median()),
        step=0.1
    )

    rad = st.number_input(
        "RAD — Highway accessibility index",
        min_value=0.0,
        value=float(df["rad"].median()),
        step=1.0
    )

    tax = st.number_input(
        "TAX — Property tax rate",
        min_value=0.0,
        value=float(df["tax"].median()),
        step=1.0
    )

    ptratio = st.number_input(
        "PTRATIO — Pupil-teacher ratio",
        min_value=0.0,
        value=float(df["ptratio"].median()),
        step=0.1
    )

    black = st.number_input(
        "BLACK — Demographic index",
        min_value=0.0,
        value=float(df["black"].median()),
        step=1.0
    )

    lstat = st.number_input(
        "LSTAT — Lower-status population (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(df["lstat"].median()),
        step=0.1
    )

    submitted = st.form_submit_button(
        "🏠 Predict House Value",
        use_container_width=True
    )


# -----------------------------
# Prediction
# -----------------------------

if submitted:

    input_data = pd.DataFrame([{
        "crim": crim,
        "zn": zn,
        "indus": indus,
        "chas": chas,
        "nox": nox,
        "rm": rm,
        "age": age,
        "dis": dis,
        "rad": rad,
        "tax": tax,
        "ptratio": ptratio,
        "black": black,
        "lstat": lstat
    }])

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")

    st.metric(
        "Estimated House Value",
        f"${prediction:.2f}k"
    )

    st.caption(
        "MEDV is expressed in thousands of US dollars "
        "in the traditional Boston Housing dataset."
    )