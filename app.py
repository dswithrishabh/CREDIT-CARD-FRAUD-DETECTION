import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("fraud_model.pkl")
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

model = load_model()

# =========================
# App UI
# =========================
st.title("💳 Credit Card Fraud Detection System")
st.write("Machine Learning powered Fraud Detection App")

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Choose Input Type")
option = st.sidebar.radio("Select Option", ["Manual Input", "Upload CSV"])

# =========================
# MANUAL INPUT
# =========================
if option == "Manual Input":

    st.subheader("Enter Transaction Details")

    amount = st.number_input("Amount", min_value=0.0)

    v1 = st.number_input("V1")
    v2 = st.number_input("V2")
    v3 = st.number_input("V3")
    v4 = st.number_input("V4")
    v5 = st.number_input("V5")

    if st.button("Predict"):

        if model is not None:
            try:
                input_data = np.array([[amount, v1, v2, v3, v4, v5]])

                prediction = model.predict(input_data)
                prob = model.predict_proba(input_data)

                if prediction[0] == 1:
                    st.error(f"⚠️ Fraud Detected (Confidence: {prob[0][1]:.2f})")
                else:
                    st.success(f"✅ Legit Transaction (Confidence: {prob[0][0]:.2f})")

            except Exception as e:
                st.error(f"Prediction error: {e}")

# =========================
# CSV UPLOAD
# =========================
elif option == "Upload CSV":

    st.subheader("Upload CSV File")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)
        st.write("Preview:", df.head())

        if st.button("Run Prediction"):

            if model is not None:
                try:
                    # REMOVE TARGET COLUMN IF EXISTS
                    df = df.drop(columns=["Class"], errors="ignore")

                    # MATCH MODEL FEATURES EXACTLY
                    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

                    # PREDICT
                    predictions = model.predict(df)
                    df["Prediction"] = predictions

                    st.success("Prediction Completed!")
                    st.write(df.head())

                    # DOWNLOAD BUTTON
                    csv = df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Results",
                        csv,
                        "fraud_predictions.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"Prediction error: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.write("Made by Rishabh Pandey 🚀")