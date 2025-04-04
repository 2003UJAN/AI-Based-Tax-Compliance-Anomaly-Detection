import streamlit as st
import pandas as pd
import plotly.express as px
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load trained AutoEncoder model
model = tf.keras.models.load_model("models/autoencoder_model.h5")

# Streamlit UI
st.set_page_config(page_title="🕵️ Tax Fraud Detector", page_icon="💰", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF4500;'>💰 AI Tax Compliance & Anomaly Detector 🚀</h1>", unsafe_allow_html=True)

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload financial transaction data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocess Data
    features = ['Amount', 'Tax_Reported', 'Tax_Actual']
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features])

    # Predict Anomalies
    reconstructions = model.predict(df_scaled)
    anomaly_scores = np.mean(np.abs(reconstructions - df_scaled), axis=1)
    
    threshold = np.percentile(anomaly_scores, 95)  # 95th percentile as threshold
    df["Anomaly_Score"] = anomaly_scores
    df["Predicted_Anomaly"] = df["Anomaly_Score"] > threshold

    # Display results
    st.write(f"🚨 **Detected {df['Predicted_Anomaly'].sum()} suspicious transactions!**")

    # Funky Anomaly Visualization
    fig = px.scatter(df, x="Transaction_ID", y="Anomaly_Score", color="Predicted_Anomaly",
                     title="🛑 Anomaly Score Distribution", color_continuous_scale="reds")
    st.plotly_chart(fig)

    st.success("✅ Anomaly Detection Completed!")

st.write("### **📊 Power BI Dashboard:** [View Here](https://app.powerbi.com/)")
