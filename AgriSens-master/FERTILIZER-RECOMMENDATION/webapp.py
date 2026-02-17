## Importing necessary libraries for the web app
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Display Image
from PIL import Image
img = Image.open("fertilizer.png")  # optional image
st.image(img)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Fertilizer_recommendation.csv")

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# Split data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# Train RandomForest
RF = RandomForestClassifier(n_estimators=100, random_state=5)
RF.fit(Xtrain, Ytrain)

# Accuracy
predicted_values = RF.predict(Xtest)
acc = accuracy_score(Ytest, predicted_values)
print("Model Accuracy:", acc)

# -----------------------------
# Save Model
# -----------------------------
RF_pkl_filename = "Fertilizer_RF.pkl"
with open(RF_pkl_filename, "wb") as file:
    pickle.dump(RF, file)

# Load model
RF_Model_pkl = pickle.load(open("Fertilizer_RF.pkl", "rb"))

# -----------------------------
# Prediction Function
# -----------------------------
def predict_fertilizer(temp, humidity, moisture, soil, crop, nitrogen, potassium, phosphorous):
    soil_encoded = label_encoders["Soil Type"].transform([soil])[0]
    crop_encoded = label_encoders["Crop Type"].transform([crop])[0]

    features = np.array([
        temp, humidity, moisture,
        soil_encoded, crop_encoded,
        nitrogen, potassium, phosphorous
    ]).reshape(1, -1)

    prediction = RF_Model_pkl.predict(features)
    fertilizer = label_encoders["Fertilizer Name"].inverse_transform(prediction)
    return fertilizer

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.markdown("<h1 style='text-align: center;'>SMART FERTILIZER RECOMMENDATIONS</h1>", unsafe_allow_html=True)

    st.sidebar.title("AgriSens")
    st.sidebar.header("Enter Soil & Crop Details")

    temp = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 50.0)
    moisture = st.sidebar.number_input("Moisture (%)", 0.0, 100.0, 30.0)

    nitrogen = st.sidebar.number_input("Nitrogen (ppm)", 0.0, 200.0, 100.0)
    potassium = st.sidebar.number_input("Potassium (ppm)", 0.0, 200.0, 100.0)
    phosphorous = st.sidebar.number_input("Phosphorous (ppm)", 0.0, 200.0, 100.0)

    soil = st.sidebar.selectbox("Soil Type", label_encoders["Soil Type"].classes_)
    crop = st.sidebar.selectbox("Crop Type", label_encoders["Crop Type"].classes_)

    if st.sidebar.button("Predict"):
        prediction = predict_fertilizer(
            temp, humidity, moisture,
            soil, crop,
            nitrogen, potassium, phosphorous
        )

        st.success(f"Recommended Fertilizer: {prediction[0]}")

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    main()
