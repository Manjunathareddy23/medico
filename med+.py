import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
from googletrans import Translator
import asyncio
import torch.nn as nn

# Load the pre-trained CheXNet model or a similar model (DenseNet for chest X-rays)
def load_model():
    try:
        # Using DenseNet121 as an example, as it's commonly used for X-ray classification
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 14)  # Modify output for 14 classes (for ChestX-ray14 dataset)
        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocessing the uploaded image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to the model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize based on pre-trained model
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Predicting diagnosis using the model
def predict_diagnosis(image, model):
    processed_image = preprocess_image(image)
    with torch.no_grad():  # No need to calculate gradients for inference
        output = model(processed_image)
        prediction = torch.nn.functional.softmax(output, dim=1)  # Apply softmax to get probabilities
        predicted_class = prediction.argmax(dim=1).item()  # Get the predicted class index
        confidence = prediction[0, predicted_class].item()  # Confidence score
    return predicted_class, confidence

# Asynchronous translation function
async def translate_text_async(text, target_language):
    translator = Translator()
    try:
        translation = await asyncio.to_thread(translator.translate, text, dest=target_language)
        return translation.text
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return text

# Function to run translation asynchronously
def translate_text(text, target_language):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(translate_text_async(text, target_language))

# Streamlit UI
st.title("X-ray Report Diagnosis System")

# Upload X-ray image
uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])

# Language selection for diagnosis translation
language = st.selectbox("Select Language", ["en", "hi", "es", "fr", "de", "zh"])

# Load the pre-trained model
model = load_model()

if model is not None:
    st.write("Model successfully loaded!")

# Process uploaded image and make predictions
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        # Check if the file is a valid image format
        if image.format not in ["JPEG", "PNG"]:
            st.error("Invalid image format. Please upload a valid JPEG or PNG image.")
        else:
            st.image(image, caption="Uploaded X-ray", use_container_width=True)  # Fixing the deprecated warning

            # Make predictions if the model is loaded
            if model is not None:
                predicted_class, confidence = predict_diagnosis(image, model)

                # Debugging: Show the raw predicted class and confidence
                st.write(f"Predicted Class Index: {predicted_class}")
                st.write(f"Confidence: {confidence * 100:.2f}%")

                # Map predicted class index to human-readable label (ChestX-ray14 classes)
                diagnosis_map = {
                    0: "Atelectasis", 1: "Cardiomegaly", 2: "Consolidation", 3: "Edema", 4: "Effusion",
                    5: "Emphysema", 6: "Fibrosis", 7: "Hernia", 8: "Infiltration", 9: "Mass",
                    10: "No Finding", 11: "Nodule", 12: "Pleural Thickening", 13: "Pneumonia"
                }

                predicted_label = diagnosis_map.get(predicted_class, "Unknown Diagnosis")
                translated_diagnosis = translate_text(f"Diagnosis: {predicted_label} (Confidence: {confidence * 100:.2f}%)", language)

                # Display results
                st.write("### Diagnosis Result:")
                st.write(translated_diagnosis)
            else:
                st.warning("Model is not loaded.")
    except Exception as e:
        st.error(f"Error processing image: {e}")
