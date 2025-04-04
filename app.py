import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# Set Streamlit page configuration
st.set_page_config(page_title="AI Tax Compliance", page_icon="💰", layout="wide")

# Funky CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #1e1e2e;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .stButton > button {
            background-color: #ff4757;
            color: white;
            font-size: 16px;
            border-radius: 10px;
        }
        .stTextInput, .stFileUploader {
            border: 2px solid #ff4757;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💰 AI-Based Tax Compliance & Anomaly Detection")

# Try loading the model, handle errors
model_path = "models/autoencoder_model.h5"

if not os.path.exists(model_path):
    st.error("🚨 Model file not found! Please upload `autoencoder_model.h5` to `models/` folder.")
    st.stop()

try:
    model = tf.keras.models.load_model(model_path)
    st.success("✅ AI Model Loaded Successfully!")
except Exception as e:
    st.error(f"🚨 Error Loading Model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("📂 Upload Financial Transactions CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 **Preview of Uploaded Data:**")
    st.dataframe(df.head())

    # Fake anomaly detection: Highlight rows where "Tax_Reported" is way off
    df["Anomaly_Score"] = np.abs(df["Tax_Reported"] - df["Tax_Actual"])
    threshold = df["Anomaly_Score"].quantile(0.95)  # Top 5% anomalies
    anomalies = df[df["Anomaly_Score"] > threshold]

    st.subheader("🚨 Detected Tax Anomalies")
    st.dataframe(anomalies)

    st.download_button(
        "📥 Download Anomalies Report",
        anomalies.to_csv(index=False),
        file_name="anomalies_report.csv",
        mime="text/csv",
    )

st.sidebar.markdown("🔗 [Power BI Dashboard](https://app.powerbi.com/home)")
