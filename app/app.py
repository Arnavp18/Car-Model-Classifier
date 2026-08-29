import json
import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from dataset import eval_transform  # noqa: E402
from train import build_model  # noqa: E402

MODEL_PATH = "models/car_classifier.pth"
CLASSES_PATH = "models/classes.json"
UNKNOWN_THRESHOLD = 0.60


@st.cache_resource
def load_model():
    with open(CLASSES_PATH) as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(classes), freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, classes, device


st.set_page_config(page_title="Car Classifier", page_icon="🚗")
st.title("🚗 Car Make & Model Classifier")
st.write("Upload a photo of a car to identify its make and model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    try:
        model, classes, device = load_model()
    except FileNotFoundError:
        st.error(
            "No trained model found yet. Train the model first with "
            "`python src/train.py`, then place the weights at "
            f"`{MODEL_PATH}` and the class list at `{CLASSES_PATH}`."
        )
        st.stop()

    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)

    top_prob, top_idx = probabilities.max(dim=0)

    if top_prob.item() < UNKNOWN_THRESHOLD:
        st.subheader("Prediction: Unknown")
        st.write("Model confidence is too low to make a reliable prediction.")
    else:
        st.subheader(f"Prediction: {classes[top_idx.item()]}")
        st.write(f"Confidence: {top_prob.item() * 100:.1f}%")

    st.write("### Top predictions")
    ranked = sorted(zip(classes, probabilities.tolist()), key=lambda x: x[1], reverse=True)
    for name, prob in ranked[:5]:
        st.write(f"{name}")
        st.progress(min(prob, 1.0))
