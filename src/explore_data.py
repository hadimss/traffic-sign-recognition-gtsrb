"""
Explore the GTSRB traffic sign dataset using TorchVision.

This script downloads the dataset, saves sample images, and creates
a class distribution plot for the training set.
"""

from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from torchvision.datasets import GTSRB


OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data/raw")

CLASS_NAMES = [
    "Speed limit 20 km/h",
    "Speed limit 30 km/h",
    "Speed limit 50 km/h",
    "Speed limit 60 km/h",
    "Speed limit 70 km/h",
    "Speed limit 80 km/h",
    "End of speed limit 80 km/h",
    "Speed limit 100 km/h",
    "Speed limit 120 km/h",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing for vehicles over 3.5 tons",
]


def main():
    print("Downloading/loading GTSRB dataset...")

    train_dataset = GTSRB(
        root=str(DATA_DIR),
        split="train",
        download=True,
    )

    test_dataset = GTSRB(
        root=str(DATA_DIR),
        split="test",
        download=True,
    )

    print("\nDataset loaded successfully.")
    print(f"Training images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Number of classes: {len(CLASS_NAMES)}")

    # Save sample image grid
    fig, axes = plt.subplots(4, 5, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(20):
        image, label = train_dataset[i]
        axes[i].imshow(image)
        axes[i].set_title(f"{label}: {CLASS_NAMES[label]}", fontsize=8)
        axes[i].axis("off")

    plt.tight_layout()
    sample_path = OUTPUT_DIR / "sample_traffic_signs.png"
    plt.savefig(sample_path, dpi=300)
    plt.close()

    print(f"Saved sample image grid to: {sample_path}")

    # Class distribution
    labels = [label for _, label in train_dataset]
    label_counts = Counter(labels)

    distribution = pd.DataFrame({
        "class_id": list(label_counts.keys()),
        "count": list(label_counts.values()),
    }).sort_values("class_id")

    plt.figure(figsize=(16, 6))
    plt.bar(distribution["class_id"].astype(str), distribution["count"])
    plt.xlabel("Traffic Sign Class ID")
    plt.ylabel("Number of Images")
    plt.title("GTSRB Training Set Class Distribution")
    plt.xticks(rotation=90)
    plt.tight_layout()

    distribution_path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(distribution_path, dpi=300)
    plt.close()

    print(f"Saved class distribution plot to: {distribution_path}")


if __name__ == "__main__":
    main()