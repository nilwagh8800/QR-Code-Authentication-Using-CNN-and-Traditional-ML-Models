import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.feature import local_binary_pattern

# Constants
IMG_SIZE = 128

# Load models
svm_model = joblib.load('traditional_model.pkl')  # Saved SVM model
cnn_model = load_model('cnn_qr_model.h5')         # Saved CNN model

# Feature extraction for traditional ML
def extract_lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

# Title
st.title("🔍 QR Code Authentication")
st.write("Upload a QR code image to classify it as **original** or **counterfeit**.")

# Model selection
model_choice = st.selectbox("Choose a model", ["Traditional ML (SVM)", "Deep Learning (CNN)"])

# File uploader
uploaded_file = st.file_uploader("Upload QR Code Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model_choice == "Traditional ML (SVM)":
        features = extract_lbp_features(image).reshape(1, -1)
        prediction = svm_model.predict(features)[0]
    else:
        resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        input_img = img_to_array(normalized).reshape(1, IMG_SIZE, IMG_SIZE, 1)
        prediction = np.argmax(cnn_model.predict(input_img), axis=1)[0]

    label = "✅ Original" if prediction == 0 else "❌ Counterfeit"
    st.subheader(f"Prediction: {label}")

