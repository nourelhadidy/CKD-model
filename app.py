import streamlit as st
import numpy as np
import joblib

# === Load model and tools
model = joblib.load("xgb_ckd_model.pkl")
scaler = joblib.load("scaler(2).pkl")
selector = joblib.load("selector(1).pkl")
selected_features = joblib.load("selected_feature_names.pkl")

# === eGFR calculation
def calculate_egfr(creatinine, age, sex):
    k = 0.7 if sex == 'female' else 0.9
    alpha = -0.329 if sex == 'female' else -0.411
    sex_factor = 1.018 if sex == 'female' else 1
    return 141 * min(creatinine / k, 1) ** alpha * max(creatinine / k, 1) ** -1.209 * 0.993 ** age * sex_factor

# === Title
st.title("Chronic Kidney Disease Prediction App")
st.write("Enter patient details to predict the likelihood of CKD.")

# === User inputs
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 100, 45)
blood_pressure = st.number_input("Blood Pressure (mmHg)", 0.0, 200.0, 80.0)
specific_gravity = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
albumin = st.slider("Albumin", 0, 5, 1)
sugar = st.slider("Sugar", 0, 5, 0)
blood_glucose_random = st.number_input("Blood Glucose Random (mg/dL)", 0.0, 500.0, 100.0)
blood_urea = st.number_input("Blood Urea (mg/dL)", 0.0, 300.0, 40.0)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.0, 15.0, 1.2)
sodium = st.number_input("Sodium (mEq/L)", 100.0, 160.0, 137.0)
potassium = st.number_input("Potassium (mEq/L)", 2.0, 8.0, 4.5)
hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 13.5)
packed_cell_volume = st.number_input("Packed Cell Volume", 20.0, 55.0, 40.0)
white_blood_cell_count = st.number_input("WBC Count (cells/cumm)", 3000.0, 18000.0, 9000.0)
red_blood_cell_count = st.number_input("RBC Count (millions/cmm)", 2.0, 6.5, 4.8)
hypertension = st.selectbox("Hypertension", ["yes", "no"])
diabetes = st.selectbox("Diabetes Mellitus", ["yes", "no"])
anemia = st.selectbox("Anemia", ["yes", "no"])
edema = st.selectbox("Edema", ["yes", "no"])

# === Predict button
if st.button("Predict CKD Risk"):

    # === Compute eGFR
    egfr = calculate_egfr(serum_creatinine, age, sex)

    # === Build input feature dictionary
    input_data = {
        "age": age,
        "blood_pressure": blood_pressure,
        "specific_gravity": specific_gravity,
        "albumin": albumin,
        "sugar": sugar,
        "blood_glucose_random": blood_glucose_random,
        "blood_urea": blood_urea,
        "serum_creatinine": serum_creatinine,
        "sodium": sodium,
        "potassium": potassium,
        "hemoglobin": hemoglobin,
        "packed_cell_volume": packed_cell_volume,
        "white_blood_cell_count": white_blood_cell_count,
        "red_blood_cell_count": red_blood_cell_count,
        "hypertension": 1 if hypertension == "yes" else 0,
        "diabetes": 1 if diabetes == "yes" else 0,
        "anemia": 1 if anemia == "yes" else 0,
        "edema": 1 if edema == "yes" else 0,
        "eGFR": egfr
    }

    # === Arrange input in the same order used during training
    full_feature_order = scaler.feature_names_in_
    full_input = [input_data[feat] for feat in full_feature_order]

    # === Scale first (on all original features)
    scaled = scaler.transform([full_input])

    # === Then select features
    selected = selector.transform(scaled)

    # === Make prediction
    prediction = model.predict(selected)[0]
    probability = model.predict_proba(selected)[0][1]

    # === Output
   # st.subheader("Estimated Glomerular Filtration Rate (eGFR)")
    st.info(f"eGFR: **{egfr:.2f} mL/min/1.73m²**")

    if egfr >= 90:
        st.success("Normal kidney function (Stage 1)")
    elif egfr >= 60:
        st.info("Mildly decreased function (Stage 2)")
    elif egfr >= 30:
        st.warning("Moderate decrease in function (Stage 3)")
    elif egfr >= 15:
        st.error("Severe decrease in function (Stage 4)")
    else:
        st.error("Kidney failure (Stage 5)")

    if prediction == 1:
        st.error(f"⚠️ High risk of Chronic Kidney Disease ({probability * 100:.2f}% probability)")
    else:
        st.success(f"✅ Low risk of Chronic Kidney Disease ({(1 - probability) * 100:.2f}% probability)")
