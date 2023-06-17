import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import os

st.title("Face Classification App")

# Load the trained model
model = load_model("face-mobile.h5")

# Get the label names from the subdirectory names under the "data" directory
root_dir = os.getcwd()
label_names = os.listdir(f"{root_dir}/data")
label_names = sorted(label_names)
print(label_names)
# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)
    return image

# Function to classify the image
def classify_image(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    class_index = np.argmax(predictions)
    class_label = label_names[class_index]
    confidence = predictions[0][class_index]
    return class_label, confidence

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify the uploaded image
    class_label, confidence = classify_image(image)

    # Display the classification result
    st.write(f"Class Label: {class_label}")
    st.write(f"Confidence: {confidence:.2f}")
