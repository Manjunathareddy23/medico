import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from googletrans import Translator

def load_model():
    # Load a pre-trained model (replace with your own model if available)
    model = tf.keras.models.load_model('xray_model.h5')
    return model

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_diagnosis(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    diagnosis = "Pneumonia" if prediction[0] > 0.5 else "Normal"
    return diagnosis

def translate_text(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

# Streamlit UI
st.title("X-ray Report Diagnosis System")

uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])
language = st.selectbox("Select Language", ["en", "hi", "es", "fr", "de", "zh"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    model = load_model()
    diagnosis = predict_diagnosis(image, model)
    translated_diagnosis = translate_text(f"Diagnosis: {diagnosis}", language)
    
    st.write("### Diagnosis Result:")
    st.write(translated_diagnosis)
