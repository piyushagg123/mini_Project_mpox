import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Inject CSS to hide the GitHub "Fork" button
hide_fork_button = """
<style>
header a[title="View source"], header a[aria-label="View source"] {
    display: none !important;
}
</style>
"""

st.markdown(hide_fork_button, unsafe_allow_html=True)
# Hide Streamlit menu and add footer
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        footer:after {
            content:'This app is in its early stage. We recommend you to seek professional advice from a dermatologist. Thank you.'; 
            visibility: visible;
            display: block;
            position: relative;
            padding: 5px;
            top: 2px;
        }
        .center { text-align: center; }
    </style>
    """, 
    unsafe_allow_html=True
)

# Load pre-trained models
model_1 = tf.keras.models.load_model('model_DenseNet121.h5')
model_2 = tf.keras.models.load_model('model_InceptionV3.h5')
model_3 = tf.keras.models.load_model('model_MobileNet.h5')

# Load optimal ensemble weights from a .npy file
weights = np.load('ensemble_weights.npy')  # This file contains weights for the ensemble

# Define labels for categories
labels = {
    0: 'Chickenpox',
    1: 'Cowpox',
    2: 'HFMD',
    3: 'Healthy',
    4: 'Measles',
    5: 'MPOX'
}

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to 224x224
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions for model input
    return image_array

# Function to make predictions using weighted ensemble
def predict(image):
    processed_image = preprocess_image(image)
    # Get predictions from all models
    pred_1 = model_1.predict(processed_image)
    pred_2 = model_2.predict(processed_image)
    pred_3 = model_3.predict(processed_image)
    
    # Weighted ensemble using weights loaded from the .npy file
    ensemble_prediction = (weights[0] * pred_1 + weights[1] * pred_2 + weights[2] * pred_3)
    
    # Final prediction
    label_index = np.argmax(ensemble_prediction)
    predicted_label = labels[label_index]
    confidence = ensemble_prediction[0][label_index] * 100
    return predicted_label, confidence

# Streamlit app
def main():
    st.markdown("<h1 class='center'>MPOX Skin Lesion Classifier</h1>", unsafe_allow_html=True)

    # Image upload options
    source = st.radio('Pick one', ['Upload from gallery', 'Capture by camera'])
    uploaded_file = st.camera_input("Take a picture") if source == 'Capture by camera' else st.file_uploader("Choose an image", type=["jpg", "png", "jpeg", "bmp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Process and classify the image
        predicted_label, confidence = predict(image)
        
        # Display results
        st.markdown(f"<h3 class='center'>This might be:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 class='center'>{predicted_label}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p class='center'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
