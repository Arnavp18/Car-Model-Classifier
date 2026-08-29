"""
Evaluation script: computes accuracy, precision, recall, F1, and a confusion matrix
for the trained car classifier.

Usage:
    python src/evaluate.py --model models/car_classifier.pth
"""

import argparse

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from dataset import get_dataloaders
from train import build_model


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/car_classifier.pth")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="assets/confusion_matrix.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, classes = get_dataloaders(args.data_dir)

    model = build_model(num_classes=len(classes), freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    labels, preds = evaluate(model, test_loader, device)

    print(classification_report(labels, preds, target_names=classes))

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"\nConfusion matrix saved to {args.output}")


if __name__ == "__main__":
    main()
