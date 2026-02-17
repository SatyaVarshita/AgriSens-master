## Importing necessary libraries for the web app
# -------------------------------
# Import Libraries
# -------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Base Directory (VERY IMPORTANT)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# Display Top Image (Cloud Safe)
# -------------------------------
image_path = os.path.join(BASE_DIR, "crop.png")

if os.path.exists(image_path):
    img = Image.open(image_path)
    st.image(img, use_column_width=True)
else:
    st.warning("crop.png not found. Please check file location.")

# -------------------------------
# Load Dataset (Cloud Safe)
# -------------------------------
csv_path = os.path.join(BASE_DIR, "Crop_recommendation.csv")

if not os.path.exists(csv_path):
    st.error("Crop_recommendation.csv not found in project folder.")
    st.stop()

df = pd.read_csv(csv_path)

# -------------------------------
# Prepare Data
# -------------------------------
X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------
# Train Model (Cached for Speed)
# -------------------------------
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=20, random_state=5)
    model.fit(Xtrain, Ytrain)
    return model

model = train_model()

# -------------------------------
# Prediction Function
# -------------------------------
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    return prediction[0]

# -------------------------------
# Streamlit UI
# -------------------------------
def main():

    st.markdown(
        "<h1 style='text-align: center;'>🌾 SMART CROP RECOMMENDATION 🌾</h1>",
        unsafe_allow_html=True
    )

    st.sidebar.title("AgriSens 🌱")
    st.sidebar.header("Enter Soil & Weather Details")

    nitrogen = st.sidebar.number_input("Nitrogen (N)", 0.0, 140.0, 0.0)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", 0.0, 145.0, 0.0)
    potassium = st.sidebar.number_input("Potassium (K)", 0.0, 205.0, 0.0)
    temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 51.0, 0.0)
    humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 0.0)
    ph = st.sidebar.number_input("pH Level", 0.0, 14.0, 0.0)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)

    if st.sidebar.button("Predict"):
        if (nitrogen == 0 and phosphorus == 0 and potassium == 0 and
            temperature == 0 and humidity == 0 and ph == 0 and rainfall == 0):

            st.error("⚠ Please enter valid values before predicting.")
        else:
            result = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            st.success(f"🌱 Recommended Crop: {result}")

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    main()
