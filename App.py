import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Load trained RandomForest GridSearchCV pipeline
# -------------------------
model = pickle.load(open('best_model_RandomForestClassifier.pkl', 'rb'))  # Make sure this file is in the same folder

# -------------------------
# Streamlit UI
# -------------------------
st.title("🫀 Heart Disease Risk Predictor")
st.write("Enter your health details below to estimate your heart disease risk:")

# User inputs
age = st.number_input("Age", 20, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1=True, 0=False)", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# Convert categorical input
sex = 1 if sex == "Male" else 0
dataset = 0  # default value used in training

# -------------------------
# Build input DataFrame
# -------------------------
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'dataset': dataset,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalch': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}])

# -------------------------
# Predict using pipeline
# -------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.error(f"⚠ High Risk of Heart Disease (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Low Risk of Heart Disease (Probability: {probability:.2f})")
    except Exception as e:
        st.error(f"Error: {e}")
