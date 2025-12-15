import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Eye Disease Detection", layout="centered")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mobilenetv2_eye_disease.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["Class1", "Class2", "Class3", "Class4"]  # replace with your labels

st.title("👁️ Eye Disease Detection System")

uploaded_file = st.file_uploader(
    "Upload a retinal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🩺 Predicted Disease: **{predicted_class}**")
