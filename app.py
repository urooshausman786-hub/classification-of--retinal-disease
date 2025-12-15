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
/* Gradient header */
h1 {
    background: linear-gradient(90deg, #36D1DC, #5B86E5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    margin-bottom: 0;
}

/* Subtitle */
h3 {
    color: #ff6347;
    text-align: center;
    margin-top: 0;
    font-weight: 500;
}

/* File uploader */
.css-1v0mbdj.edgvbvh3 {
    border: 2px dashed #5B86E5;
    border-radius: 10px;
    padding: 20px;
    background-color: #f0f8ff;
}

/* Center app */
.stApp {
    max-width: 900px;
    margin: auto;
}

/* Prediction result */
.prediction {
    font-size: 24px;
    font-weight: bold;
    color: #2E8B57;
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
    "📤 Drag & drop a retinal image here or click to browse (JPG, PNG, max 200MB)",
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
