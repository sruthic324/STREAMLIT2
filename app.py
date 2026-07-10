import pickle
import pandas as pd
import streamlit as st

# Load model into file
with open("model.pkl","rb") as f:
    model = pickle.load(f)

# Try loading scaler (if exit)
try:
    with open("scaler.pkl","rb") as f:
        scaler=pickle.load(f)
except:
    scaler= None

# Prediction function

def preprocess_and_predict(features):

# convert to Dataframe 
    input_data=pd.DataFrame([features])

# Get required columns
    required_columns=model.feature_names_in_
   
#Add missing columns
    for col in required_columns:
      if col not in input_data.columns:
         input_data[col]=0

#Arrange columns
    input_data=input_data[required_columns]
   
#Apply scaling only if scaler exists
    if scaler is not None:
       input_data=scaler.transform(input_data)

#Prediction
    prediction=model.predict(input_data)
    probability=model.predict_proba(input_data)[:,1]
    return prediction[0], probability[0]

## UI

st.title("Heart Disese Prediction App") 
age =st.number_input("Age", 1,120,50)
sex = st.selectbox("Sex", ["Male", "Female"]) 
sex = 1 if sex== "Male" else 0
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps =st.number_input("Resting Blood Pressure", 80, 200, 120)
chol=st.number_input("Cholesterol", 100,600,200)
fbs = st.selectbox("Fasting", [0,1,2])
restecg = st.selectbox("Resting ECG", [0,1,2])
thalach = st.number_input("Max Heart Rate", 60, 220,150) 
exang = st.selectbox ("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("ST Depressioon",0.0,6.0,1.0, step=0.1) 
slope=st.selectbox("slope", [0,1,2])
ca = st.selectbox ("Major Vessels", [0,1,2,3,4])
thal = st.selectbox ("Thalassemia", [0,1,2,3])

# input dictionary
features= {
    "age":age,
    "sex":sex,
    "cp":cp,
    "trestbps":trestbps,
    "chol":chol,
    "fbs":fbs,
    "restecg":restecg,
    "thalach":thalach,
    "exang":exang,
    "oldpeak":oldpeak,
    "slope":slope,
    "ca":ca,
    "thal":thal

}

# Prediction
if st.button("Predict"):
    prediction, probability = preprocess_and_predict(features)
    if prediction == 1:
        st.error(f"high chnace of heart disease(probalility:{probability:.2f})")
    else:
        st.success(f"low chnace of heart disease(probalility:{probability:.2f})")