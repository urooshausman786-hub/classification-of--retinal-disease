import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="🩺 Retinal Disease Classification",
    page_icon="👁️",
    layout="wide"
)

# -----------------------------
# Custom CSS for MSc-level UI
# -----------------------------
st.markdown("""
<style>
/* Gradient background for whole app */
.stApp {
    background: linear-gradient(to bottom right, #36D1DC, #5B86E5);
    color: #ffffff; /* Default text color */
    max-width: 900px;
    margin: auto;
    position: relative;
    min-height: 100vh;
}

/* Watermark background image */
.stApp::before {
    content: "";
    background: url('https://i.imgur.com/1G1uV4Y.png') no-repeat center;
    background-size: 200px 200px;  /* Adjust size */
    opacity: 0.05;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    position: absolute;
    width: 100%;
    height: 100%;
    z-index: 0;
}

/* Gradient header */
h1 {
    background: linear-gradient(90deg, #36D1DC, #5B86E5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    margin-bottom: 0;
    position: relative;
    z-index: 1;
}

/* Subtitle */
h3 {
    color: #ffd700;  /* Golden subtitle */
    text-align: center;
    margin-top: 0;
    font-weight: 500;
    position: relative;
    z-index: 1;
}

/* File uploader */
.css-1v0mbdj.edgvbvh3 {
    border: 2px dashed #ffffff;
    border-radius: 10px;
    padding: 20px;
    background-color: rgba(255, 255, 255, 0.1); /* semi-transparent */
    color: #ffffff;
    position: relative;
    z-index: 1;
}

/* Prediction result */
.prediction {
    font-size: 24px;
    font-weight: bold;
    color: #ffff00;  /* Bright yellow for emphasis */
    position: relative;
    z-index: 1;
}

/* Adjust progress bar color */
.stProgress > div > div > div {
    background-color: #ffd700 !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.title("🩺 Retinal Disease Classification System")
st.markdown("<h3>Upload a retinal image to detect possible eye diseases using CNN (MobileNetV2)</h3>", unsafe_allow_html=True)

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Drag & drop a retinal image here or click to browse (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Retinal Image", use_column_width=True)

    st.info("👁️ Running model inference...")

    # -----------------------------
    # Load tflite model
    # -----------------------------
    interpreter = tf.lite.Interpreter(model_path="mobilenetv2_eye_disease.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Predicted class and confidence
    classes = ["Normal", "Diabetic Retinopathy", "Glaucoma", "AMD"]
    pred_class = np.argmax(output_data)
    confidence = output_data[pred_class]

    # Display results in columns
    col1, col2 = st.columns([2,1])
    with col1:
        st.success(f"✅ Predicted Disease: **{classes[pred_class]}**")
    with col2:
        st.progress(int(confidence*100))
        st.write(f"Confidence: {confidence*100:.2f}%")

else:
    st.info("👁️ Please upload a retinal image to begin diagnosis.")
