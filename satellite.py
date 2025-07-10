import streamlit as st
import numpy as np
import os
import requests
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ----------------- Model Setup -------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_PATH = "Modelenv.v1.h5"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (only once)..."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return load_model(MODEL_PATH)

model = download_and_load_model()
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# ------------------ UI Setup ----------------------
st.set_page_config(page_title="Satellite Image Classifier 🌍", layout="centered")

st.title("🛰️ Satellite Image Classifier")
st.markdown("Upload a satellite image below to classify it into one of the following categories:")
st.markdown("🔹 **Cloudy**  \n🔹 **Desert**  \n🔹 **Green Area**  \n🔹 **Water**")

uploaded_file = st.file_uploader("📤 Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((256, 256))
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # ------------------ Preprocessing ----------------------
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ------------------ Prediction -------------------------
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown("## ✅ Prediction:")
    st.success(f"**{predicted_class}** with {confidence * 100:.2f}% confidence.")

    # ------------------ Plotly Chart -----------------------
    fig = go.Figure(go.Bar(
        x=prediction,
        y=class_names,
        orientation='h',
        marker=dict(color='skyblue'),
        hovertemplate='%{y}: %{x:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title="📊 Class Confidence Scores",
        xaxis=dict(title='Confidence', range=[0, 1]),
        yaxis=dict(title='Class'),
        template='simple_white',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)
