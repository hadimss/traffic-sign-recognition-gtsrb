"""
Streamlit application for Traffic Sign Recognition.
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import get_device, load_trained_model, predict_image


MODEL_PATH = ROOT_DIR / "models" / "best_resnet18_gtsrb.pth"


st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered",
)


@st.cache_resource
def load_model():
    device = get_device()
    model = load_trained_model(
        model_path=str(MODEL_PATH),
        device=device,
    )
    return model, device


st.title("🚦 Traffic Sign Recognition")
st.write(
    "Upload a cropped traffic sign image and the model will predict "
    "the traffic sign class using a fine-tuned ResNet-18 model."
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a traffic sign image",
    type=["jpg", "jpeg", "png", "ppm"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, caption="Input traffic sign image", use_container_width=True)

    try:
        model, device = load_model()

        predictions = predict_image(
            image=image,
            model=model,
            device=device,
            top_k=3,
        )

        top_prediction, top_confidence = predictions[0]

        st.subheader("Prediction")
        st.success(f"{top_prediction}")
        st.metric("Confidence", f"{top_confidence * 100:.2f}%")

        st.subheader("Top 3 Predictions")

        for rank, (class_name, confidence) in enumerate(predictions, start=1):
            st.write(f"**{rank}. {class_name}** — {confidence * 100:.2f}%")
            st.progress(float(confidence))

    except FileNotFoundError:
        st.error(
            "Trained model checkpoint was not found. "
            "Please make sure `models/best_resnet18_gtsrb.pth` exists locally."
        )

else:
    st.info("Upload an image to start prediction.")

st.markdown("---")
st.caption(
    "Model: Fine-tuned ResNet-18 | Dataset: GTSRB | Task: Traffic Sign Classification"
)