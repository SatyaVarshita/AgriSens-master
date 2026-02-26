import streamlit as st
import tensorflow as tf
import numpy as np
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])
#app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# import Image from pillow to open images
import os
from PIL import Image

# Get directory where main.py exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full path to image
image_path = os.path.join(BASE_DIR, "Diseases.png")

img = Image.open(image_path)
# display image using streamlit
# width is used to set the width of an image
st.image(img)

#Main Page
if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION", unsafe_allow_html=True)
    
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
                # ==========================================
        # Complete Fertilizer & Pesticide Suggestions
        # ==========================================

        recommendation = {

        'Apple___Apple_scab':{
            "fertilizer":"Apply balanced NPK (10-10-10) and increase potassium.",
            "pesticide":"Spray Captan or Mancozeb fungicide every 7 days."
        },
        'Apple___Black_rot':{
            "fertilizer":"Apply compost rich in organic matter and potassium.",
            "pesticide":"Use Myclobutanil fungicide spray."
        },
        'Apple___Cedar_apple_rust':{
            "fertilizer":"Use balanced NPK and micronutrient foliar spray.",
            "pesticide":"Apply Sulfur or Copper-based fungicide."
        },
        'Apple___healthy':{
            "fertilizer":"Maintain regular compost and balanced NPK.",
            "pesticide":"No pesticide required."
        },

        'Blueberry___healthy':{
            "fertilizer":"Use acidic fertilizer (Ammonium sulfate).",
            "pesticide":"No pesticide required."
        },

        'Cherry_(including_sour)___Powdery_mildew':{
            "fertilizer":"Apply balanced NPK and avoid excess nitrogen.",
            "pesticide":"Spray Sulfur fungicide weekly."
        },
        'Cherry_(including_sour)___healthy':{
            "fertilizer":"Apply compost and balanced fertilizer.",
            "pesticide":"No pesticide required."
        },

        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':{
            "fertilizer":"Use nitrogen-rich fertilizer moderately.",
            "pesticide":"Apply Azoxystrobin fungicide."
        },
        'Corn_(maize)___Common_rust_':{
            "fertilizer":"Apply balanced NPK fertilizer.",
            "pesticide":"Spray Propiconazole fungicide."
        },
        'Corn_(maize)___Northern_Leaf_Blight':{
            "fertilizer":"Increase potassium levels.",
            "pesticide":"Apply Mancozeb fungicide."
        },
        'Corn_(maize)___healthy':{
            "fertilizer":"Use nitrogen and phosphorus fertilizer regularly.",
            "pesticide":"No pesticide required."
        },

        'Grape___Black_rot':{
            "fertilizer":"Apply compost and potassium-rich fertilizer.",
            "pesticide":"Spray Myclobutanil fungicide."
        },
        'Grape___Esca_(Black_Measles)':{
            "fertilizer":"Apply organic compost and micronutrients.",
            "pesticide":"Use Thiophanate-methyl fungicide."
        },
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':{
            "fertilizer":"Balanced NPK fertilizer.",
            "pesticide":"Apply Copper fungicide."
        },
        'Grape___healthy':{
            "fertilizer":"Apply compost and potassium.",
            "pesticide":"No pesticide required."
        },

        'Orange___Haunglongbing_(Citrus_greening)':{
            "fertilizer":"Use micronutrient spray (Zinc, Iron, Manganese).",
            "pesticide":"Control psyllids using Imidacloprid insecticide."
        },

        'Peach___Bacterial_spot':{
            "fertilizer":"Balanced NPK with calcium supplement.",
            "pesticide":"Spray Copper-based bactericide."
        },
        'Peach___healthy':{
            "fertilizer":"Apply compost and balanced fertilizer.",
            "pesticide":"No pesticide required."
        },

        'Pepper,_bell___Bacterial_spot':{
            "fertilizer":"Use nitrogen-rich fertilizer moderately.",
            "pesticide":"Apply Copper fungicide or Streptomycin."
        },
        'Pepper,_bell___healthy':{
            "fertilizer":"Balanced NPK fertilizer.",
            "pesticide":"No pesticide required."
        },

        'Potato___Early_blight':{
            "fertilizer":"Apply potassium-rich fertilizer.",
            "pesticide":"Use Chlorothalonil fungicide."
        },
        'Potato___Late_blight':{
            "fertilizer":"Balanced NPK fertilizer.",
            "pesticide":"Spray Metalaxyl or Mancozeb."
        },
        'Potato___healthy':{
            "fertilizer":"Use nitrogen and phosphorus fertilizer.",
            "pesticide":"No pesticide required."
        },

        'Raspberry___healthy':{
            "fertilizer":"Apply compost and balanced fertilizer.",
            "pesticide":"No pesticide required."
        },

        'Soybean___healthy':{
            "fertilizer":"Apply phosphorus-rich fertilizer.",
            "pesticide":"No pesticide required."
        },

        'Squash___Powdery_mildew':{
            "fertilizer":"Avoid excess nitrogen fertilizer.",
            "pesticide":"Spray Sulfur or Potassium bicarbonate."
        },

        'Strawberry___Leaf_scorch':{
            "fertilizer":"Apply compost and potassium.",
            "pesticide":"Use Captan fungicide."
        },
        'Strawberry___healthy':{
            "fertilizer":"Balanced NPK fertilizer.",
            "pesticide":"No pesticide required."
        },

        'Tomato___Bacterial_spot':{
            "fertilizer":"Balanced fertilizer with calcium.",
            "pesticide":"Spray Copper bactericide."
        },
        'Tomato___Early_blight':{
            "fertilizer":"Nitrogen-rich fertilizer.",
            "pesticide":"Apply Chlorothalonil."
        },
        'Tomato___Late_blight':{
            "fertilizer":"Balanced NPK fertilizer.",
            "pesticide":"Spray Metalaxyl."
        },
        'Tomato___Leaf_Mold':{
            "fertilizer":"Apply potassium-rich fertilizer.",
            "pesticide":"Use Mancozeb fungicide."
        },
        'Tomato___Septoria_leaf_spot':{
            "fertilizer":"Balanced fertilizer.",
            "pesticide":"Apply Copper fungicide."
        },
        'Tomato___Spider_mites Two-spotted_spider_mite':{
            "fertilizer":"Apply organic compost.",
            "pesticide":"Use Neem oil or Abamectin."
        },
        'Tomato___Target_Spot':{
            "fertilizer":"Balanced NPK fertilizer.",
            "pesticide":"Spray Azoxystrobin."
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus':{
            "fertilizer":"Apply micronutrient spray.",
            "pesticide":"Control whiteflies using Imidacloprid."
        },
        'Tomato___Tomato_mosaic_virus':{
            "fertilizer":"Apply balanced fertilizer and compost.",
            "pesticide":"Remove infected plants; disinfect tools."
        },
        'Tomato___healthy':{
            "fertilizer":"Balanced NPK fertilizer.",
            "pesticide":"No pesticide required."
        }

        }

        predicted_class = class_name[result_index]

        fert = recommendation[predicted_class]["fertilizer"]
        pest = recommendation[predicted_class]["pesticide"]

        st.subheader("🌱 Fertilizer Recommendation")
        st.info(fert)

        st.subheader("🧪 Pesticide Recommendation")
        st.warning(pest)
