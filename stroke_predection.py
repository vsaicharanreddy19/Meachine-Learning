# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 09:56:13 2025

@author: DELL
"""

import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st

warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv("C:Documents/Downloads/Brolly_internship/Stroke/healthcare-dataset-stroke-data.csv")

# Data preprocessing
data["bmi"].fillna(np.mean(data["bmi"]), inplace=True)
data["gender"] = data["gender"].replace({"Male": 1, "Female": 0, "Other": 2}).astype(int)
data["ever_married"] = data["ever_married"].replace({"Yes": 1, "No": 0}).astype(int)
data["work_type"] = data["work_type"].replace({"Private": 1, "Self-employed": 2, "Govt_job": 3, "children": 4, "Never_worked": 5}).astype(int)
data["Residence_type"] = data["Residence_type"].replace({"Urban": 1, "Rural": 0}).astype(int)

# One-hot encoding for smoking status
data_encoded = pd.get_dummies(data["smoking_status"], drop_first=True).astype(int)
main = pd.concat([data, data_encoded], axis=1).drop(columns=["smoking_status", "work_type"])

# Define features and target variable
X = main.drop(columns=["stroke"])
y = main["stroke"]

# Handle class imbalance using SMOTETomek
smk = SMOTETomek(random_state=42)
X_resampled, y_resampled = smk.fit_resample(X, y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.5, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
stroke_predictor = LogisticRegression()
stroke_predictor.fit(X_train_scaled, y_train)
y_pred = stroke_predictor.predict(X_test_scaled)

# Evaluate model accuracy
accuracy = accuracy_score(y_pred, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and scaler
pickle.dump(stroke_predictor, open('stroke_predictor.sav', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Streamlit App for Stroke Prediction
st.sidebar.title("Stroke Prediction App")
st.sidebar.markdown("Provide the patient's details below:")

# Collect user input from sidebar
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
age = st.sidebar.slider("Age", 0, 120, 50)
hypertension = st.sidebar.selectbox("Hypertension", options=[0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", options=[0, 1])
ever_married = st.sidebar.selectbox("Ever Married", options=["Yes", "No"])
residence_type = st.sidebar.selectbox("Residence Type", options=["Urban", "Rural"])
avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
smoking_status = st.sidebar.selectbox("Smoking Status", options=["formerly smoked", "never smoked", "smokes", "Unknown"])

# Encode user input
gender_encoded = 1 if gender == "Male" else 0
ever_married_encoded = 1 if ever_married == "Yes" else 0
residence_encoded = 1 if residence_type == "Urban" else 0
smoking_encoded = [0, 0, 0]
if smoking_status == "formerly smoked":
    smoking_encoded[0] = 1
elif smoking_status == "never smoked":
    smoking_encoded[1] = 1
elif smoking_status == "smokes":
    smoking_encoded[2] = 1

# Prepare input data for prediction
input_data = [[gender_encoded, age, hypertension, heart_disease, ever_married_encoded, residence_encoded, avg_glucose_level, bmi] + smoking_encoded]
input_df = pd.DataFrame(input_data, columns=["gender", "age", "hypertension", "heart_disease", "ever_married", "Residence_type", "avg_glucose_level", "bmi", "formerly smoked", "never smoked", "smokes"])

# Load model and scaler
with open('stroke_predictor.sav', "rb") as model_file:
    stroke_predictor = pickle.load(model_file)
with open('scaler.pkl', "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Prediction button
if st.sidebar.button("Predict Stroke"):
    scaled_input = scaler.transform(input_df)
    prediction = stroke_predictor.predict(scaled_input)
    probability = stroke_predictor.predict_proba(scaled_input)
    
    # Display results
    st.write("### Prediction Results")
    st.write("**Stroke Risk Prediction:**", "Yes" if prediction[0] == 1 else "No")
    st.write("**Probability of Stroke:**", f"{probability[0][1] * 100:.2f}%")

# Display app instructions
st.markdown("""
- This app predicts the likelihood of stroke based on patient details.
- Adjust inputs using the sliders and dropdowns in the sidebar.
""")
