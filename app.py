# app.py  — Minimal, runnable Streamlit template (no external model required)

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# -----------------------------
# Basic page config
# -----------------------------
st.set_page_config(page_title="Wheat Yield Prediction (Minimal)", layout="wide")
st.title("Wheat Yield Prediction (Minimal Template)")

REQUIRED_COLS = ["temperature", "humidity", "wind", "cluster"]  # numeric cluster: 0/1/2/3


# -----------------------------
# Model loading / fallback
# -----------------------------
@st.cache_resource
def load_or_build_model():
    """
    Try to load model_global.pkl; if missing, train a tiny baseline model on synthetic data.
    Returns a fitted sklearn Pipeline.
    """
    model_path = "model_global.pkl"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model, "loaded_from_file"
        except Exception as e:
            st.warning(f"Failed to load model_global.pkl, using fallback model. Detail: {e}")

    # Fallback model: fit on synthetic data (fast and small)
    rng = np.random.RandomState(42)
    n = 400

    temp = rng.uniform(10, 40, size=n)          # °C
    humi = rng.uniform(10, 95, size=n)          # %
    wind = rng.uniform(0, 40, size=n)           # km/h
    clus = rng.randint(0, 4, size=n)            # 0..3
    # A simple synthetic target: linear-ish relation + noise
    y = 0.08*temp + 0.015*humi - 0.03*wind + 0.5*clus + rng.normal(0, 0.3, size=n)

    X = np.vstack([temp, humi, wind, clus]).T
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("linreg", LinearRegression())
    ])
    pipe.fit(X, y)
    return pipe, "fallback_trained"


model, model_source = load_or_build_model()
st.caption(f"Model source: {model_source}")


# -----------------------------
# Utilities
# -----------------------------
def make_df_from_inputs(temperature, humidity, wind, cluster):
    return pd.DataFrame(
        [[float(temperature), float(humidity), float(wind), int(cluster)]],
        columns=REQUIRED_COLS
    )

def predict_df(df: pd.DataFrame) -> np.ndarray:
    # Ensure column order and numeric type
    df = df.copy()
    df = df[REQUIRED_COLS].apply(pd.to_numeric, errors="coerce")
    if df[REQUIRED_COLS].isna().any().any():
        raise ValueError("Found non-numeric values in required columns.")
    X = df.values
    return model.predict(X)


# -----------------------------
# Sidebar: mode selection
# -----------------------------
st.sidebar.header("Prediction Mode")
mode = st.sidebar.radio("Choose:", ["Single Input", "Batch Upload"])


# -----------------------------
# Session history
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=REQUIRED_COLS + ["prediction"])


# -----------------------------
# Single input mode
# -----------------------------
def ui_single():
    st.subheader("Single Prediction")

    c1, c2 = st.columns(2)
    with c1:
        temperature = st.number_input("Temperature (°C)", min_value= -20.0, max_value= 60.0, value=25.0, step=0.1)
        humidity    = st.slider("Humidity (%)", min_value=0, max_value=100, value=60, step=1)
    with c2:
        wind        = st.slider("Wind Speed (km/h)", min_value=0, max_value=80, value=10, step=1)
        cluster     = st.selectbox("Climate Cluster", options=[0, 1, 2, 3], index=0)

    if st.button("Predict"):
        try:
            row = make_df_from_inputs(temperature, humidity, wind, cluster)
            yhat = predict_df(row)[0]
            st.success(f"Predicted yield: {yhat:.3f} tons/ha")

            # append to history
            row["prediction"] = yhat
            st.session_state.history = pd.concat([st.session_state.history, row], ignore_index=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # history view and download
    if not st.session_state.history.empty:
        st.markdown("#### Prediction History")
        st.dataframe(st.session_state.history, use_container_width=True)
        csv_bytes = st.session_state.history.to_csv(index=False).encode("utf-8")
        st.download_button("Download History CSV", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv")


# -----------------------------
# Batch upload mode
# -----------------------------
@st.cache_data
def template_csv_bytes():
    tmp = pd.DataFrame(
        {"temperature": [25.0], "humidity": [60], "wind": [10], "cluster": [0]}
    )
    buf = io.StringIO()
    tmp.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def ui_batch():
    st.subheader("Batch Prediction")
    st.download_button("Download Template CSV", data=template_csv_bytes(), file_name="template.csv", mime="text/csv")

    up = st.file_uploader("Upload CSV with required columns: "
                          + ", ".join(REQUIRED_COLS),
                          type=["csv"])
    if up is None:
        return

    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    try:
        preds = predict_df(df)
        out = df.copy()
        out["prediction"] = np.round(preds, 3)
        st.success("Batch prediction finished.")
        st.dataframe(out, use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", data=csv_bytes, file_name="batch_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")


# -----------------------------
# Entry
# -----------------------------
if mode == "Single Input":
    ui_single()
else:
    ui_batch()
