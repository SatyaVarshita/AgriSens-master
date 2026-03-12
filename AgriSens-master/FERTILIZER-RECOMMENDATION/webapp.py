# -----------------------------
# Import Libraries
# -----------------------------
import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Get Project Directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, "Fertilizer_recommendation.csv")
img_path = os.path.join(BASE_DIR, "fertilizer.png")

# -----------------------------
# Display Image
# -----------------------------
if os.path.exists(img_path):
    img = Image.open(img_path)
    st.image(img)
else:
    st.warning("fertilizer.png image not found")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(csv_path)

# -----------------------------
# Encode Categorical Data
# -----------------------------
label_encoders = {}

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# Prepare Features
# -----------------------------
X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# -----------------------------
# Train Model
# -----------------------------
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, y, test_size=0.3, random_state=42
)

RF_Model = RandomForestClassifier(n_estimators=100, random_state=5)
RF_Model.fit(Xtrain, Ytrain)

# Accuracy
accuracy = accuracy_score(Ytest, RF_Model.predict(Xtest))

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

    prediction = RF_Model.predict(features)

    fertilizer = label_encoders["Fertilizer Name"].inverse_transform(prediction)

    return fertilizer[0]

# -----------------------------
# Streamlit UI
# -----------------------------
def main():

    st.markdown(
        "<h1 style='text-align:center;'>🌱 Smart Fertilizer Recommendation System</h1>",
        unsafe_allow_html=True
    )

    st.write(f"Model Accuracy: **{round(accuracy*100,2)}%**")

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

    if st.sidebar.button("Predict Fertilizer"):

        result = predict_fertilizer(
            temp,
            humidity,
            moisture,
            soil,
            crop,
            nitrogen,
            potassium,
            phosphorous
        )

        st.success(f"✅ Recommended Fertilizer: **{result}**")

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    main()
