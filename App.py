import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load("Heart Disease Detector")  # Ensure your model is in the same directory or adjust path

# App title and description
st.set_page_config(page_title="Heart Disease Predictor", layout="centered", page_icon="❤️")

st.markdown("""
    <h2 style='text-align: center;'>❤️ Heart Disease Prediction App</h2>
    <p style='text-align: center;'>Enter the patient's details to predict the risk of heart disease.</p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar for navigation and info
with st.sidebar:
    st.title("ℹ️ Info")
    st.markdown("""
        This app uses a machine learning model to predict whether a patient is at risk of heart disease.
        
        **Features used**:
        - Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, etc.

        Built with ❤️ using Streamlit and Scikit-learn.
    """)
    theme = st.selectbox("🎨 Choose Theme", ["Light", "Dark"])

    if theme == "Dark":
        st.markdown("<style>body { background-color: #1e1e1e; color: white; }</style>", unsafe_allow_html=True)

# Input form
with st.form("input_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 45)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
        chol = st.number_input("Cholesterol (chol)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved (thalach)", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope of ST Segment (slope)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]
    result = "🟢 No Heart Disease" if prediction == 0 else "🔴 High Risk of Heart Disease"

    st.success(f"Prediction: {result}")

    st.markdown(f"""
        <div style='text-align: center; font-size: 18px; padding-top: 10px;'>
            <b>Model Prediction:</b> {result}
        </div>
    """, unsafe_allow_html=True)