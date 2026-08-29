"""
Single-image prediction for the car make/model classifier.

Usage:
    python src/predict.py --image path/to/car.jpg --model models/car_classifier.pth
"""

import argparse
import json

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import eval_transform
from train import build_model

UNKNOWN_THRESHOLD = 0.60


def load_classes(path="models/classes.json"):
    with open(path) as f:
        return json.load(f)


def predict_image(image_path, model, transform, device, classes):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)

    top_prob, top_idx = probabilities.max(dim=0)

    if top_prob.item() < UNKNOWN_THRESHOLD:
        prediction = "Unknown"
    else:
        prediction = classes[top_idx.item()]

    ranked = sorted(
        zip(classes, probabilities.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return prediction, top_prob.item(), ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/car_classifier.pth")
    parser.add_argument("--classes", type=str, default="models/classes.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.classes)

    model = build_model(num_classes=len(classes), freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    prediction, confidence, ranked = predict_image(
        args.image, model, eval_transform, device, classes
    )

    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence * 100:.1f}%\n")
    print("Top predictions:")
    for name, prob in ranked[:5]:
        print(f"  {name:20s} {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
