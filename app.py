import streamlit as st
import joblib

# Load trained model
model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Classification")

st.write("Enter the measurements of the Iris flower:")

# Input fields
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2
)

# Prediction button
if st.button("Predict"):
    sample = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(sample)

    st.success(f"Predicted Species: {prediction[0]}")
