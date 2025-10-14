# 🫀 Heart Disease Prediction App

This interactive web application predicts the likelihood of **heart disease** using a trained **Random Forest Classifier**.  
It was developed as part of my applied research in medical data analysis and machine learning — integrating data preprocessing, model optimization, and deployment into an intuitive Streamlit interface.

---

## 🚀 Features
- Predicts the presence of heart disease based on clinical data.
- Machine learning pipeline built with **Python**, **scikit-learn**, and **XGBoost**.
- Web interface developed using **Streamlit** for interactive predictions.
- Deployed on **Streamlit Cloud** for easy public access.
- Scalable and easily extendable for future healthcare applications.

---

## 🧠 How It Works
1. The machine learning model (`best_model_RandomForestClassifier.pkl`) was trained using patient data with features like **age, cholesterol, blood pressure, resting ECG results**, and others.
2. Users input medical parameters into the app.
3. The app preprocesses the inputs, applies scaling, and feeds the data into the trained model.
4. The model outputs a prediction indicating whether the patient is **at risk of heart disease or not**.

---

## 🧩 Installation & Local Setup
```bash
# Clone this repository
git clone https://github.com/AbdoulieJBah/heart-disease-predictor.git
cd heart-disease-predictor

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run App.py
