import streamlit as st
from joblib import load
import numpy as np

# Load the saved Random Forest model
rf_model = load("random_forest_model.joblib")

# Create a Streamlit web app
st.title("Diabetes Prediction App")

# Sidebar for user input
st.sidebar.header("Enter Patient Information")

# Create input fields for each feature
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=17, value=0)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=0)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=122, value=0)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=99, value=0)
weight_kg = st.sidebar.number_input("Weight (kg)", min_value=0.0, value=0.0)
height_cm = st.sidebar.number_input("Height (cm)", min_value=0.0, value=0.0)

# Ask for related parameters for Diabetes Pedigree Function
family_history = st.sidebar.radio("Family History of Diabetes", ["Yes", "No"])
pedigree_age = st.sidebar.number_input("Age of Family Member", min_value=0, value=0)

# Calculate Diabetes Pedigree Function (DPF) based on family history and age
if family_history == "Yes":
    dpf = pedigree_age / 50  # You can adjust the calculation method as needed
else:
    dpf = 0.0

# Create a button to predict
if st.sidebar.button("Predict"):
    # Prepare the input features as a NumPy array
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, weight_kg, height_cm, dpf]).reshape(1, -1)

    # Use the loaded model to make predictions
    prediction = rf_model.predict(input_data)

    # Display the prediction result
    if prediction[0] == 1:
        st.error("The patient is likely to have diabetes.")
    else:

        st.success("The patient is likely not to have diabetes.")

# Optionally, you can provide some context and instructions
st.write("\n")
st.write("Input the patient's information in the sidebar, including family history and age, and click 'Predict' to see the result.")
