import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle
from PIL import Image

# Load Model and Class Indices
def load_model():
    model = tf.keras.models.load_model("./models/crop_disease_model.h5")
    return model

def load_class_indices():
    with open("./models/class_indices.pkl", "rb") as f:
        class_indices = pickle.load(f)
    return {v: k for k, v in class_indices.items()}  # Reverse mapping

# Preprocess Image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict Function
def predict_image(img, model, class_labels):
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    class_idx = np.argmax(predictions)
    return class_labels[class_idx]

# Load model and class labels
model = load_model()
class_labels = load_class_indices()

# Streamlit UI
st.title("🌾 Crop Disease Detection App")
st.write("Upload an image of a crop leaf, and the model will predict its disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_file = Image.open(uploaded_file)
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict Disease"):
        prediction = predict_image(image_file, model, class_labels)
        st.success(f"🩺 Predicted Disease: **{prediction}**")
