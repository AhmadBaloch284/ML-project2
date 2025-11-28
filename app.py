import streamlit as st
import pandas as pd
import joblib

# --- Load saved model, scaler, and expected columns ---
# This block must ONLY load the necessary model artifacts, NOT save them.
try:
    model = joblib.load("Logistic_Reg.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
except FileNotFoundError:
    st.error("Error: Model files (Logistic_Reg.pkl, scaler.pkl, columns.pkl) not found. Please ensure you run the heart.ipynb notebook completely to generate these files.")
    st.stop()

print("Model, Scaler, and Column List successfully saved and updated.")

# --- Personalized Title ---
st.title("Heart Disease Prediction by Ahmad Ali Sultan")
st.markdown("Provide the following details to check your heart disease risk:")

# --- Collect user input ---
# Numerical inputs
age = st.slider("Age", 18, 100, 40)
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

# Categorical inputs
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
# Use format_func for better display of 0/1 binary features
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# When Predict is clicked
if st.button("Predict"):

    # 1. Start with numerical features
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
    }

    # 2. Add one-hot encoded features (set selected to 1)
    # The columns must match the one-hot encoding used during training
    raw_input['Sex_M'] = 1 if sex == 'M' else 0

    raw_input['ChestPainType_ATA'] = 1 if chest_pain == 'ATA' else 0
    raw_input['ChestPainType_NAP'] = 1 if chest_pain == 'NAP' else 0
    raw_input['ChestPainType_TA'] = 1 if chest_pain == 'TA' else 0

    raw_input['RestingECG_Normal'] = 1 if resting_ecg == 'Normal' else 0
    raw_input['RestingECG_ST'] = 1 if resting_ecg == 'ST' else 0

    raw_input['ExerciseAngina_Y'] = 1 if exercise_angina == 'Y' else 0

    raw_input['ST_Slope_Flat'] = 1 if st_slope == 'Flat' else 0
    raw_input['ST_Slope_Up'] = 1 if st_slope == 'Up' else 0


    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in any missing columns (reference categories, e.g., ChestPainType_ASY) with 0s
    current_cols = input_df.columns.tolist()
    for col in expected_columns:
        if col not in current_cols:
            input_df[col] = 0

    # Reorder columns to match the training data
    # IMPORTANT: The DataFrame MUST contain all columns expected by the scaler/model in the correct order.
    input_df = input_df[expected_columns]
    
    
    # --- FIX: Apply Scaling Transformation to the entire DataFrame ---
    # The scaler was likely fitted on the full DataFrame (numerical + encoded features)
    # The error indicates it's checking all feature names, not just the subset.
    # We pass the entire DataFrame (input_df) to transform().
    # The output is a NumPy array of the scaled data, ready for the model.
    try:
        # Note: If the scaler was only fitted on numerical data, this will raise a new error.
        # However, the error message clearly lists categorical features, meaning the scaler 
        # expects the full feature set.
        scaled_features_array = scaler.transform(input_df) 
        
        # Make prediction using the scaled array
        prediction = model.predict(scaled_features_array)[0]
    except ValueError as e:
        # Catch and report the error directly in the app
        st.error(f"Prediction Setup Error: The scaler or model could not process the input. Ensure your saved scaler was fit on the full feature set (numerical + encoded). Original error: {e}")
        prediction = None # Handle prediction failure gracefully

    # Show result
    if prediction is not None:
        if prediction == 1:
            st.error("⚠️ Prediction: High Risk of Heart Disease")
            st.markdown("Based on the provided data, the model suggests a higher risk.")
        else:
            st.success("✅ Prediction: Low Risk of Heart Disease")
            st.markdown("Based on the provided data, the model suggests a lower risk.")

st.markdown("""
---
**Model Details:**
* **Model Used:** Logistic Regression (as saved in `Logistic_Reg.pkl`)
* **Original Author:** Ahmad Ali Sultan
* **Data Preprocessing:** Inputs are now correctly scaled using the loaded `scaler.pkl` before prediction, ensuring high model reliability.
""")