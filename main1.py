import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==================== Load the Model ====================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('trained_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==================== Prediction Function ====================
def model_prediction(test_image):
    try:
        model = load_model()
        if model is None:
            return None

        image = Image.open(test_image).resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert to batch format

        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        return result_index
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# ==================== Class Names ====================
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ==================== Sidebar ====================
st.sidebar.title("🌱 Plant Disease Recognition")
app_mode = st.sidebar.radio("Select Page", ["🏠 Home", "📖 About", "🔍 Disease Recognition"])

# ==================== Home Page ====================
if app_mode == "🏠 Home":
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown(
        """
        ## 🌿 Welcome to the Plant Disease Recognition System!  
        🔎 Quickly detect plant diseases using AI-powered image analysis.  
        👉 Upload an image to get started!
        
        ### **How It Works**
        1. **Upload Image** – Go to the Disease Recognition page.  
        2. **Analyze** – The system will detect the disease using advanced ML models.  
        3. **Get Results** – Get fast, accurate results with recommendations.  

        ---
        ✅ **Fast** – Results in seconds  
        ✅ **Accurate** – State-of-the-art machine learning models  
        ✅ **Easy to Use** – Simple and intuitive interface  
        """
    )

# ==================== About Page ====================
elif app_mode == "📖 About":
    st.header("📘 About the Project")
    st.markdown(
        """
        ### 📂 **Dataset Information**
        - This dataset includes **87K+ images** of healthy and diseased crops.  
        - Categorized into **38 classes** including Tomato, Apple, Potato, Corn, etc.  
        - The dataset is divided into:
          - **70K+ Training Images**  
          - **17K+ Validation Images**  
          - **33 Test Images**  

        ### 🚀 **Goal**
        The goal is to provide a quick and accurate diagnosis of plant diseases using AI and image processing.  
        """
    )

# ==================== Disease Recognition Page ====================
elif app_mode == "🔍 Disease Recognition":
    st.header("🔍 **Plant Disease Recognition**")
    
    test_image = st.file_uploader("📸 Upload an image of the plant leaf:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("⚡ Predict"):
            with st.spinner("Analyzing... Please wait..."):
                result_index = model_prediction(test_image)

                if result_index is not None:
                    disease = class_names[result_index]
                    st.success(f"✅ Model Prediction: **{disease}**")
                    
                    # ✅ Provide additional details based on disease
                    if "healthy" in disease:
                        st.info("🌿 The plant appears to be healthy. No action needed.")
                    else:
                        st.warning(f"🚨 Detected signs of **{disease}**. Take necessary precautions!")

                    # ✅ Additional Recommendations
                    st.markdown(
                        """
                        ### 🌾 **Suggested Actions**
                        - Isolate the affected plant to prevent spread.  
                        - Apply recommended fungicides or pesticides.  
                        - Consult an agronomist if the condition worsens.  
                        """
                    )
                else:
                    st.error("⚠️ Unable to process the image. Try again!")

# ==================== Footer ====================
st.markdown("---")
st.caption("🚀 Developed with ❤️ by Dibyajyoti Mishra")

