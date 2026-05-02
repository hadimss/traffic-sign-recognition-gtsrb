"""
Evaluate the trained ResNet-18 model on the GTSRB test set.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from dataset import CLASS_NAMES, create_dataloaders
from model import build_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_model(model, dataloader, device):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return all_labels, all_predictions


def save_confusion_matrix(labels, predictions, output_path):
    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(18, 18))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(range(len(CLASS_NAMES))),
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=90)

    plt.title("GTSRB Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    model_path = Path("models/best_resnet18_gtsrb.pth")
    report_dir = Path("reports")
    figure_dir = report_dir / "figures"

    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train the model first."
        )

    device = get_device()
    print(f"Using device: {device}")

    print("Loading test data...")
    _, _, test_loader = create_dataloaders(
        image_size=224,
        batch_size=32,
        num_workers=2,
    )

    print("Loading trained model...")
    checkpoint = torch.load(model_path, map_location=device)

    model = build_model(num_classes=43)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    labels, predictions = evaluate_model(model, test_loader, device)

    accuracy = accuracy_score(labels, predictions)

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "test_accuracy": accuracy,
        "model_path": str(model_path),
        "num_test_samples": len(labels),
    }

    metrics_path = report_dir / "test_metrics.json"
    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    report_df = pd.DataFrame(report).transpose()
    report_path = report_dir / "classification_report.csv"
    report_df.to_csv(report_path)

    confusion_matrix_path = figure_dir / "confusion_matrix.png"
    save_confusion_matrix(labels, predictions, confusion_matrix_path)

    print("\nEvaluation complete.")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Saved test metrics to: {metrics_path}")
    print(f"Saved classification report to: {report_path}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")


if __name__ == "__main__":
    main()