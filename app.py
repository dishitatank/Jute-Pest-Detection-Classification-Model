import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

MODEL_PATH = "mobilenet_model.h5"  # If in the same directory as app.py

# Load the trained model
model = load_model(MODEL_PATH)

# Define the class labels
class_labels = [
    "Beet Armyworm", "Black Hairy", "Cutworm", "Field Cricket", "Jute Aphid", 
    "Jute Hairy", "Jute Red Mite", "Jute Semilooper", "Jute Stem Girdler", 
    "Jute Stem Weevil", "Leaf Beetle", "Mealybug", "Pod Borer", 
    "Scopula Emissaria", "Termite", "Termite odontotermes (Rambur)", "Yellow Mite"
]

# Confidence threshold
CONFIDENCE_THRESHOLD = 50.0  # Percentage

# Streamlit App UI
st.title("Jute Pest Detection using MobileNet")
st.write("Upload An Image of Pest to Identify it's Type.")

# Image uploader
uploaded_file = st.file_uploader("Choose An Image", type=["jpg", "jpeg", "png"])

# Prediction logic
if uploaded_file is not None:
    try:
        # Load and preprocess the image
        img = Image.open(uploaded_file)
        img_resized = img.resize((150, 150))  # Resize to match training input size
        img_array = np.array(img_resized) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])  # Get the class with the highest probability
        confidence = predictions[0][class_index] * 100  # Confidence in percentage

        # Display results
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        if confidence >= CONFIDENCE_THRESHOLD:
            st.success(f"**Predicted Pest:** {class_labels[class_index]}")
            st.info(f"**Prediction Confidence:** {confidence:.2f}%")
        else:
            st.warning("The model is not confident about the prediction. Please upload a clearer image of a pest.")

    except Exception as e:
        st.error(f"An Error Occurred During Processing: {e}")
else:
    st.info("Please Upload An Image To Proceed.")
