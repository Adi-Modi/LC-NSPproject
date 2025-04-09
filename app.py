import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.bin')

# App Configuration
st.set_page_config(page_title="Liver Disease Predictor", page_icon="🩺", layout="centered")

# Custom styling
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        color: #1f2937;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .title {
        color: #2b6cb0;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .dark-mode {
        background-color: #1e1e2f !important;
        color: #f3f4f6 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🩺 Liver Disease Prediction</div>", unsafe_allow_html=True)

# Theme toggle using session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

if st.button("🌓 Toggle Theme"):
    st.session_state.dark_mode = not st.session_state.dark_mode

if st.session_state.dark_mode:
    st.markdown("""
        <style>
        .stApp {
            background-color: #1e1e2f;
            color: #f3f4f6;
        }
        </style>
    """, unsafe_allow_html=True)

# Input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", options=["Female", "Male"])
    bilirubin = st.number_input("Bilirubin (mg/dL)", min_value=0.0, step=0.1, value=1.2)
    albumin = st.number_input("Albumin (g/dL)", min_value=0.0, step=0.1, value=3.2)
    prothrombin = st.number_input("Prothrombin Time (seconds)", min_value=0.0, step=0.1, value=10.5)
    submit = st.form_submit_button("🔍 Predict")

# Prediction logic
if submit:
    try:
        sex_val = 1.0 if sex == "Male" else 0.0

        # Construct input (match the shape and order expected by model)
        X_input = np.array([[0.0, age, sex_val, 0.0, 0.0, 0.0, 0.0,
                             bilirubin, 300.0, albumin, 80.0, 1000.0,
                             60.0, 120.0, 250.0, prothrombin]])

        X_scaled = scaler.transform(X_input)
        pred = model.predict(X_scaled)[0]

        if pred == 1:
            st.success("✅ The person has a normal liver.")
        elif pred == 2:
            st.warning("⚠️ The person may have fatty liver.")
        elif pred == 3:
            st.error("❗ The person may have liver fibrosis.")
        elif pred == 4:
            st.error("🚨 The person may have liver cirrhosis.")
        else:
            st.info("🤔 Unexpected prediction result.")
    except Exception as e:
        st.error(f"Error in prediction: {e}")