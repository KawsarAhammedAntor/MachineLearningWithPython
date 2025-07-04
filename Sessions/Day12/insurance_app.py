import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model using a path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'insurance_model.pkl')
model = joblib.load(model_path)

st.set_page_config(page_title="Insurance Charges Predictor", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f0f4f8;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .result-box {background-color: #d4edda; color: #155724; border-radius: 10px; padding: 20px; font-size: 22px; font-weight: bold;}
    .input-box {background-color: #e3e3ff; border-radius: 10px; padding: 20px;}
    </style>
""", unsafe_allow_html=True)

st.title("💡 Insurance Charges Predictor")
st.write("Enter the details below to predict insurance charges.")

col1, col2 = st.columns([2, 2])

with col1:
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    age = st.slider('Age', 18, 100, 30)
    sex = st.selectbox('Sex', ['female', 'male'])
    bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)
    children = st.slider('Number of Children', 0, 5, 0)
    smoker = st.selectbox('Smoker', ['no', 'yes'])
    region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if st.button('Predict Charges'):
        # Prepare input for model
        input_dict = {
            'age': age,
            'sex': 1 if sex == 'male' else 0,
            'bmi': bmi,
            'children': children,
            'smoker': 1 if smoker == 'yes' else 0,
            'region_northeast': 1 if region == 'northeast' else 0,
            'region_northwest': 1 if region == 'northwest' else 0,
            'region_southeast': 1 if region == 'southeast' else 0,
            'region_southwest': 1 if region == 'southwest' else 0,
        }
        input_df = pd.DataFrame([input_dict])
        pred = model.predict(input_df)[0]
        st.markdown(f'<div class="result-box">Predicted Charges: ${pred:,.2f}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box" style="background:#f8d7da;color:#721c24;">Prediction will appear here.</div>', unsafe_allow_html=True)
