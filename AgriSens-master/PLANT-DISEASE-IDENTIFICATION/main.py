import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------- LOAD MODEL SAFELY -------------------- #
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")
        return model
    except Exception as e:
        st.error("Model failed to load. Please check model file and TensorFlow version.")
        st.stop()

model = load_model()

# -------------------- CLASS NAMES -------------------- #
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# -------------------- FERTILIZER RECOMMENDATION -------------------- #
fertilizer_recommendation = {

    'Apple___Apple_scab': "Apply Sulfur or Captan fungicide. Use balanced NPK (10-10-10).",
    'Apple___Black_rot': "Apply Mancozeb fungicide. Add Potassium-rich fertilizer.",
    'Apple___Cedar_apple_rust': "Use Copper fungicide spray. Maintain balanced NPK.",
    'Apple___healthy': "No disease detected. Maintain organic compost and balanced NPK.",

    'Blueberry___healthy': "Maintain acidic soil. Use Ammonium sulfate fertilizer.",

    'Cherry_(including_sour)___Powdery_mildew': "Apply Sulfur fungicide. Use Nitrogen in controlled amount.",
    'Cherry_(including_sour)___healthy': "Maintain compost and balanced fertilizer.",

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Apply Azoxystrobin fungicide. Use Nitrogen fertilizer.",
    'Corn_(maize)___Common_rust_': "Use Mancozeb fungicide. Apply balanced NPK.",
    'Corn_(maize)___Northern_Leaf_Blight': "Apply Propiconazole fungicide. Use Nitrogen-rich fertilizer.",
    'Corn_(maize)___healthy': "Apply Nitrogen fertilizer regularly.",

    'Grape___Black_rot': "Apply Myclobutanil fungicide. Use Potassium-rich fertilizer.",
    'Grape___Esca_(Black_Measles)': "Remove infected vines. Apply balanced NPK fertilizer.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Use Copper fungicide spray. Apply Nitrogen fertilizer.",
    'Grape___healthy': "Maintain compost and balanced nutrients.",

    'Orange___Haunglongbing_(Citrus_greening)': "Apply Zinc and Iron micronutrients. Use balanced NPK fertilizer.",

    'Peach___Bacterial_spot': "Apply Copper-based bactericide. Use balanced NPK.",
    'Peach___healthy': "Maintain compost and proper irrigation.",

    'Pepper,_bell___Bacterial_spot': "Apply Copper bactericide. Use Potassium-rich fertilizer.",
    'Pepper,_bell___healthy': "Use balanced NPK fertilizer.",

    'Potato___Early_blight': "Apply Chlorothalonil fungicide. Add Nitrogen fertilizer.",
    'Potato___Late_blight': "Apply Metalaxyl fungicide. Use balanced NPK.",
    'Potato___healthy': "Apply compost and balanced fertilizer.",

    'Raspberry___healthy': "Use organic compost and balanced NPK.",

    'Soybean___healthy': "Apply Phosphorus-rich fertilizer.",

    'Squash___Powdery_mildew': "Apply Sulfur fungicide. Use balanced NPK.",

    'Strawberry___Leaf_scorch': "Apply Myclobutanil fungicide. Use Potassium-rich fertilizer.",
    'Strawberry___healthy': "Apply compost and balanced nutrients.",

    'Tomato___Bacterial_spot': "Apply Copper bactericide. Use balanced NPK.",
    'Tomato___Early_blight': "Apply Mancozeb fungicide. Add Potassium fertilizer.",
    'Tomato___Late_blight': "Apply Metalaxyl fungicide immediately.",
    'Tomato___Leaf_Mold': "Apply Chlorothalonil fungicide. Maintain balanced nutrients.",
    'Tomato___Septoria_leaf_spot': "Apply Copper fungicide. Use balanced NPK.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Apply Miticide spray. Use Nitrogen fertilizer carefully.",
    'Tomato___Target_Spot': "Apply Azoxystrobin fungicide. Use balanced NPK.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Use resistant varieties. Apply Zinc micronutrients.",
    'Tomato___Tomato_mosaic_virus': "Remove infected plants. Use balanced NPK fertilizer.",
    'Tomato___healthy': "Apply compost and balanced fertilizer regularly."
}
# -------------------- PREDICTION FUNCTION -------------------- #
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# -------------------- SIDEBAR -------------------- #
st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# -------------------- DISPLAY IMAGE -------------------- #
try:
    img = Image.open("Diseases.png")
    st.image(img)
except:
    pass

# -------------------- HOME PAGE -------------------- #
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

# -------------------- DISEASE RECOGNITION PAGE -------------------- #
elif app_mode == "DISEASE RECOGNITION":

    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an Image")

    if test_image is not None:

        st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("### Prediction Result")

            result_index = model_prediction(test_image)
            predicted_disease = class_name[result_index]

            st.success(f"Model Prediction: {predicted_disease}")

            st.subheader("Recommended Fertilizer and Pesticides")

            if predicted_disease in fertilizer_recommendation:
                st.info(fertilizer_recommendation[predicted_disease])
            else:
                st.info("Apply balanced NPK fertilizer and consult agricultural expert.")

    else:
        st.warning("Please upload an image before prediction.")
