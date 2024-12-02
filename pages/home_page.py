import streamlit as st
import numpy as np
import pickle

def run():
    # Load model and scaler
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    st.title("Diabetes Prediction")
    st.markdown("Provide your details below to predict the likelihood of diabetes.")

    # Input form
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1)
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    high_bp = st.radio("High Blood Pressure", ["Yes", "No"])
    high_chol = st.radio("High Cholesterol", ["Yes", "No"])
    gen_hlth = st.slider("General Health (1=Poor, 5=Excellent)", 1, 5)
    phys_hlth = st.number_input("Physical Health (days)", min_value=0, max_value=30, step=1)
    ment_hlth = st.number_input("Mental Health (days)", min_value=0, max_value=30, step=1)
    smoker = st.radio("Smoker", ["Yes", "No"])

    # Predict button
    if st.button("Predict"):
        input_data = np.array([[bmi, age, high_bp == "Yes", high_chol == "Yes", gen_hlth, phys_hlth, ment_hlth, smoker == "Yes"]])
        input_data_scaled = scaler.transform(input_data)
        y_pred_prob = model.predict_proba(input_data_scaled)[:, 1]
        probability = y_pred_prob[0] * 100

        st.markdown("### Prediction Result")
        if probability >= 75:
            st.error(f"High Risk: {probability:.2f}%")
        elif 50 <= probability < 75:
            st.warning(f"Moderate Risk: {probability:.2f}%")
        else:
            st.success(f"Low Risk: {probability:.2f}%")

