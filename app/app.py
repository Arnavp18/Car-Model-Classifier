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
    """Load the trained model and class list. Cached so it only runs once per session."""
    with open(CLASSES_PATH) as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(classes), freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, classes, device


def predict(image, model, classes, device):
    """Run inference on a single PIL image and return prediction, confidence, and ranked list."""
    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)

    top_prob, top_idx = probabilities.max(dim=0)
    prediction = classes[top_idx.item()] if top_prob.item() >= UNKNOWN_THRESHOLD else "Unknown"

    ranked = sorted(zip(classes, probabilities.tolist()), key=lambda x: x[1], reverse=True)

    return prediction, top_prob.item(), ranked


st.set_page_config(page_title="Car Classifier", page_icon="🚗")
st.title("🚗 Car Make & Model Classifier")
st.write("Upload a photo of a car to identify its make and model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        model, classes, device = load_model()
    except FileNotFoundError as e:
        st.error(
            "No trained model found yet. Train the model first with "
            "`python src/train.py`, save the class list to "
            f"`{CLASSES_PATH}`, and make sure both files are committed "
            f"or present in `models/`.\n\nMissing file: `{e.filename}`"
        )
        st.stop()
    except Exception as e:
        st.error(f"Something went wrong loading the model: {e}")
        st.stop()

    with st.spinner("Classifying..."):
        prediction, confidence, ranked = predict(image, model, classes, device)

    if prediction == "Unknown":
        st.subheader("Prediction: Unknown")
        st.write("Model confidence is too low to make a reliable prediction.")
    else:
        st.subheader(f"Prediction: {prediction}")
        st.write(f"Confidence: {confidence * 100:.1f}%")

    st.write("### Top predictions")
    for name, prob in ranked[:5]:
        st.write(name)
        st.progress(min(max(prob, 0.0), 1.0))
else:
    st.info("Upload a JPG or PNG image of a car to get started.")
