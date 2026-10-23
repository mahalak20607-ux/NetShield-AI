
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ids_model.joblib"
DATA_PATH = BASE_DIR / "demo_network_traffic.csv"

FEATURES = [
    "duration",
    "packet_count",
    "bytes_per_second",
    "destination_port",
    "failed_connections",
]

st.set_page_config(
    page_title="AI Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ NetShield AI")
st.subheader("Network Intrusion Detection System")
st.caption(
    "Machine Learning-based analysis of network-flow records "
    "for educational and demonstration purposes."
)

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Model or demo data is missing. Run train_model.py first.")
    st.code(r".\.venv\Scripts\python.exe train_model.py")
    st.stop()

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

st.sidebar.header("Controls")
st.sidebar.write("Choose a network-flow data source.")

source = st.sidebar.radio(
    "Data source",
    ["Demo dataset", "Upload CSV"],
)

if source == "Demo dataset":
    data = pd.read_csv(DATA_PATH)
else:
    uploaded = st.sidebar.file_uploader(
        "Upload network traffic CSV",
        type=["csv"],
    )
    if uploaded is None:
        st.info("Upload a CSV containing all required feature columns.")
        st.write("Required columns:", ", ".join(FEATURES))
        st.stop()

    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

missing = [column for column in features if column not in data.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

for column in features:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.dropna(subset=features).copy()

if data.empty:
    st.warning("No valid numeric network-flow records found.")
    st.stop()

if (data[features] < 0).any().any():
    st.error("Feature values must not be negative.")
    st.stop()

data["Prediction"] = model.predict(data[features])
data["Status"] = data["Prediction"].map({
    "Normal": "Normal",
    "Suspicious": "Potentially Suspicious",
})

total = len(data)
alerts = int((data["Prediction"] == "Suspicious").sum())
normal = total - alerts
alert_rate = alerts / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flows Analysed", f"{total:,}")
c2.metric("Normal Predictions", f"{normal:,}")
c3.metric("Suspicious Predictions", f"{alerts:,}")
c4.metric("Suspicious Rate", f"{alert_rate:.1f}%")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Traffic Classification")
    counts = (
        data["Status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Count")
    )
    fig = px.pie(
        counts,
        names="Status",
        values="Count",
        hole=0.45,
        title="Normal vs Potentially Suspicious",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Traffic Rate Distribution")
    fig = px.histogram(
        data,
        x="bytes_per_second",
        color="Status",
        nbins=30,
        title="Bytes per Second",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Potentially Suspicious Records")
alerts_data = data[data["Prediction"] == "Suspicious"]
if alerts_data.empty:
    st.success("No records were classified as suspicious in this dataset.")
else:
    st.dataframe(alerts_data.head(100), use_container_width=True)

st.subheader("All Analysed Records")
st.dataframe(data.head(500), use_container_width=True)

st.download_button(
    "Download Analysis Report (CSV)",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="intrusion_detection_report.csv",
    mime="text/csv",
)

st.caption(
    "Educational prototype: predictions are based on simulated training data. "
    "A suspicious prediction is not proof of an attack."
)
