# ❤️ Heart Disease Prediction System

### Machine Learning Web App for Early Heart Risk Assessment

A Machine Learning–powered web application built with **:contentReference[oaicite:1]{index=1}** that predicts the risk of heart disease using a trained **Logistic Regression** model. Users enter basic medical information and instantly receive a risk assessment.

---

## 📌 Project Overview

This project demonstrates a complete **end-to-end ML pipeline** — from model training and preprocessing to deployment as an interactive web app.

The app loads:
- A trained Logistic Regression model  
- A fitted scaler  
- Expected feature columns  

to ensure accurate real-time predictions.

---

## ✨ Key Features

- Heart disease risk prediction (High / Low)  
- Interactive Streamlit user interface  
- Proper feature scaling before inference  
- One-hot encoding for categorical inputs  
- Pre-trained model loaded using Joblib  
- Clean and beginner-friendly structure  

---

## 🛠 Tech Stack

**Python** • **Pandas** • **Scikit-learn** • **Streamlit** • **Joblib**

---

## ⚙ How It Works

1. User enters medical details (age, BP, cholesterol, heart rate, etc.)
2. Categorical features are one-hot encoded
3. Input data is aligned with training columns
4. Features are scaled using the saved scaler
5. Logistic Regression model predicts heart disease risk
6. Result is displayed instantly in the web interface

---

## 📊 Model Details

- Algorithm: Logistic Regression  
- Preprocessing: Feature scaling + one-hot encoding  
- Saved artifacts:
  - `Logistic_Reg.pkl`
  - `scaler.pkl`
  - `columns.pkl`

---

## 🔮 Future Improvements

- Add probability score display  
- Improve UI/UX  
- Add model comparison (Random Forest, XGBoost)  
- Deploy to cloud  
- Add patient history tracking  

---

### ⭐ Learning Outcome

This project helped me gain hands-on experience in:

- Medical ML prediction systems  
- Feature engineering & preprocessing  
- Model serialization  
- Real-world ML deployment  
- Building interactive ML apps  

---

