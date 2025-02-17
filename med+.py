import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from googletrans import Translator

# Load model function with error handling
def load_model():
    try:
        model = tf.keras.models.load_model('xray_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocessing the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predicting diagnosis
def predict_diagnosis(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    # Assuming the model outputs a single value (probability of pneumonia)
    diagnosis = "Pneumonia" if prediction[0] > 0.5 else "Normal"
    confidence = max(prediction[0], 1 - prediction[0])  # Confidence score
    return diagnosis, confidence

# Translate text once using Translator
translator = Translator()

def translate_text(text, target_language):
    try:
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return text

# Streamlit UI
st.title("X-ray Report Diagnosis System")

uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])
language = st.selectbox("Select Language", ["en", "hi", "es", "fr", "de", "zh"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        # Load model and make predictions
        model = load_model()
        if model:
            diagnosis, confidence = predict_diagnosis(image, model)
            translated_diagnosis = translate_text(f"Diagnosis: {diagnosis} (Confidence: {confidence * 100:.2f}%)", language)
            
            # Display results
            st.write("### Diagnosis Result:")
            st.write(translated_diagnosis)
        else:
            st.write("Model could not be loaded.")
    except Exception as e:
        st.error(f"Error processing image: {e}")
