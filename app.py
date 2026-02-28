import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Page configuration for a dynamic, premium look
st.set_page_config(
    page_title="EAV Sentiment Analyzer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom premium CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        color: #f1f1f1;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #e0e0ff !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(37, 117, 252, 0.4);
        border: none;
        color: white;
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: fadeIn 0.8s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .neg-emotion {
        color: #ff4757;
        font-size: 2rem;
        font-weight: bold;
    }
    .non-neg-emotion {
        color: #2ed573;
        font-size: 2rem;
        font-weight: bold;
    }
    /* Streamlit overrides for premium feel */
    div[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    div.stSlider > div[data-baseweb="slider"] > div {
        background: #2575fc;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🧠 EAV Sentiment Analyzer")
st.markdown("### Multimodal Emotion Recognition (EEG + Video)")
st.markdown("Predict whether a user is experiencing **Negative Emotion** based on their EEG signal and facial video motion variances.")

# Load Model
@st.cache_resource
def load_model():
    model_path = "models/model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("Model not found! Please run `python src/export_model.py` to generate the `model.joblib` file.")
    st.stop()

# Sidebar Inputs
st.sidebar.header("⚙️ Feature Inputs")
st.sidebar.markdown("Use sliders to input signal and motion variances.")

# EEG Inputs
st.sidebar.subheader("🧠 EEG Features")
eeg_alpha = st.sidebar.slider("Alpha Mean", -3.0, 3.0, 0.0, step=0.1)
eeg_beta = st.sidebar.slider("Beta Mean", -3.0, 3.0, 0.0, step=0.1)
eeg_theta = st.sidebar.slider("Theta Mean", -3.0, 3.0, 0.0, step=0.1)
eeg_corr_mean = st.sidebar.slider("Correlation Mean", 0.0, 1.0, 0.5, step=0.01)
eeg_corr_var = st.sidebar.slider("Correlation Variance", 0.0, 0.5, 0.1, step=0.01)

# Video Inputs
st.sidebar.subheader("📹 Video Features")
mouth_motion = st.sidebar.slider("Mouth Motion Variance", 0.0, 1.0, 0.5, step=0.01)
eye_motion = st.sidebar.slider("Eye Motion Variance", 0.0, 1.0, 0.5, step=0.01)
head_motion = st.sidebar.slider("Head Motion Variance", 0.0, 1.0, 0.5, step=0.01)

# Main Prediction Section
st.markdown("---")

if st.button("🔮 Analyze Emotion"):
    with st.spinner("Processing multimodal features..."):
        # Create Dataframe for prediction
        input_data = pd.DataFrame({
            "eeg_alpha_mean": [eeg_alpha],
            "eeg_beta_mean": [eeg_beta],
            "eeg_theta_mean": [eeg_theta],
            "eeg_corr_mean": [eeg_corr_mean],
            "eeg_corr_var": [eeg_corr_var],
            "mouth_motion_var": [mouth_motion],
            "eye_motion_var": [eye_motion],
            "head_motion_var": [head_motion],
        })

        # Predict
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        st.markdown("### Result:")

        if prediction == 1:
            st.markdown("<div class='neg-emotion'>⚠️ Negative Emotion Detected</div>", unsafe_allow_html=True)
            st.markdown(f"Confidence: **{probabilities[1]*100:.1f}%**")
            st.info("The fused EEG and Video patterns strongly indicate negative sentiment.")
        else:
            st.markdown("<div class='non-neg-emotion'>✅ Non-Negative Emotion Detected</div>", unsafe_allow_html=True)
            st.markdown(f"Confidence: **{probabilities[0]*100:.1f}%**")
            st.success("The subject appears to be in a non-negative emotional state.")

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed using Scikit-Learn Pipeline and Streamlit • [Vishesh Kaushal's GitHub](https://github.com/vishesh-2306)")
