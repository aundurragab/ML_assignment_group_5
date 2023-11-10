import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load the XGBoost model
#model = xgb.Booster()
#model.load_model('ML_Group_xgBoost_Smoking_model.pkl')

with open('ML_Group_xgBoost_Smoking_model.pkl', 'rb') as f:
    model = pickle.load(f)


def transform_data(data):
    # Transform continuous age into age bucket
    data['age_bucket'] = pd.cut(data['age'], bins=range(20, 130, 5), right=False)
    
    # Define a function to map continuous age to age bucket
    def map_age_to_bucket(age):
        return int((age // 5) * 5)

    # Apply the transformation to the 'age' column
    data['age'] = data['age'].apply(map_age_to_bucket)
    data.drop('age_bucket', axis=1, inplace=True)

    # Calculate 'heightXhemoglobin' without scaling
    data['heightXhemoglobin'] = data['height(cm)'] * data['hemoglobin']

    return data


# Function to predict using the XGBoost model
def predict_smoker(data):
    data = np.array(data).reshape(1, -1)
    dmatrix = xgb.DMatrix(data)
    prediction = model.predict(dmatrix)
    return prediction[0]

# Streamlit app
st.title("Smoking Prediction")
st.write("Tobacco kills up to 8 million people annually worlwide.\
    Smoking causes a range of desease like cancer, strokes and several lung and heart diseases.\
    Smoking also increases risk for tuberculosis, certain eye diseases, and problems of the immune system, including rheumatoid arthritis.\
    Since smoking leads to such a vast number of health problems, these problems are easily visible in a persons health data.\
    This app let's you predict if a patient is a smoker based on a few, simple health metrics.")
st.write()

st.write("*The default inputs are the average values of non-smokers.*")

col1, col2 = st.columns(2)  # Split the page into two columns for better visualisation

# First column of input form
with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=45)
    height = st.number_input("Height (cm)", min_value=120, max_value=250, value=161)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=63)
    waist = st.number_input("Waist (cm)", min_value=50, max_value=200, value=80)
    eyesight_left = st.number_input("Eyesight (left)", min_value=0, max_value=10, value=1)
    eyesight_right = st.number_input("Eyesight (right)", min_value=0, max_value=10, value=1)
    hearing_left = st.selectbox("Hearing (left)", [1, 2], index=0)
    hearing_right = st.selectbox("Hearing (right)", [1, 2], index=0)
    systolic = st.number_input("Systolic", min_value=0, max_value=250, value=120)
    relaxation = st.number_input("Relaxation", min_value=0, max_value=150, value=75)
    fasting_blood_sugar = st.number_input("Fasting Blood Sugar", min_value=0, max_value=500, value=98)

with col2:
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=197)
    triglyceride = st.number_input("Triglyceride", min_value=0, max_value=600, value=113)
    hdl = st.number_input("HDL", min_value=0, max_value=500, value=59)
    ldl = st.number_input("LDL", min_value=0, max_value=2000, value=116)
    hemoglobin = st.number_input("Hemoglobin", min_value=0, max_value=25, value=14)
    serum_creatinine = st.number_input("Serum Creatinine", min_value=0, max_value=12, value=1)
    ast = st.number_input("AST", min_value=0, max_value=1500, value=25)
    alt = st.number_input("ALT", min_value=0, max_value=3500, value=25)
    gtp = st.number_input("GTP", min_value=0, max_value=1000, value=30)
    dental_caries = st.selectbox("Dental Caries", [0, 1], index=0)  # Dropdown menu

# User input dictionary
user_input = {
    'age': age,
    'height(cm)': height,
    'weight(kg)': weight,
    'waist(cm)': waist,
    'eyesight(left)': eyesight_left,
    'eyesight(right)': eyesight_right,
    'hearing(left)': hearing_left,
    'hearing(right)': hearing_right,
    'systolic': systolic,
    'relaxation': relaxation,
    'fasting blood sugar': fasting_blood_sugar,
    'Cholesterol': cholesterol,
    'triglyceride': triglyceride,
    'HDL': hdl,
    'LDL': ldl,
    'hemoglobin': hemoglobin,
    'serum creatinine': serum_creatinine,
    'AST': ast,
    'ALT': alt,
    'Gtp': gtp,
    'dental caries': dental_caries,
}

# Transform user input
transformed_input = transform_data(pd.DataFrame([user_input]))

# Prediction
if st.button("Predict"):
    prediction = predict_smoker(transformed_input)
    st.write(f"**The patient is a {'Smoker' if prediction > 0.5 else 'Non-Smoker'}**"+".")
