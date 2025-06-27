import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from sklearn.preprocessing import StandardScaler

# Background setup
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("titanic_bg.png")

# Load model
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load raw data (for age scaling)
df_raw = pd.read_csv("train.csv")
age_mean = df_raw['Age'].mean()
age_std = df_raw['Age'].std()

# App title
st.title("🚢 Titanic Survival Prediction")

# Input form
with st.form(key="user_info"):
    age = st.number_input("Enter your Age:", min_value=1, max_value=100)
    gender = st.selectbox("Choose your Gender:", ["Male", "Female"])
    portname = st.selectbox("Choose the Port of Embarkation:", ["Cherbourg", "Queenstown", "Southampton"])
    pclass = st.selectbox("Choose Travel Class:", ["1st Class", "2nd Class", "3rd Class"])
    sibsp = st.number_input("Number of Siblings/Spouses Aboard:", min_value=0, max_value=10)
    parch = st.number_input("Number of Parents/Children Aboard:", min_value=0, max_value=10)
    submit_button = st.form_submit_button(label="Predict Survival")

if submit_button:
    sex = 0 if gender == "Male" else 1
    pclass = {"1st Class": 1, "2nd Class": 2, "3rd Class": 3}[pclass]
    embarked_Q = 1 if portname == "Queenstown" else 0
    embarked_S = 1 if portname == "Southampton" else 0
    embarked_C = 1 if portname == "Cherbourg" else 0  

    scaled_age = (age - age_mean) / age_std

    input_df = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': scaled_age,
        'SibSp': sibsp,
        'Parch': parch,
        'Embarked_Q': embarked_Q,
        'Embarked_S': embarked_S
    }])

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    # Output
    if prediction == 1:
        st.success(f"🎉 The passenger is likely to survive! Probability: {probability:.2f}%")
    else:
        st.error(f"💀 The passenger is unlikely to survive. Probability: {probability:.2f}%")