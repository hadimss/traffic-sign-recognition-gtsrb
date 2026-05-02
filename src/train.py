"""
Train a fine-tuned ResNet-18 model on the GTSRB traffic sign dataset.
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from dataset import create_dataloaders
from model import build_model


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(
            loss=running_loss / total,
            accuracy=correct / total,
        )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predictions = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(
                loss=running_loss / total,
                accuracy=correct / total,
            )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def plot_training_history(history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training and Validation Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    config = load_config()

    image_size = config["data"]["image_size"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    num_classes = config["num_classes"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    model_dir = Path(config["paths"]["model_dir"])
    report_dir = Path(config["paths"]["report_dir"])
    figure_dir = report_dir / "figures"

    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print("Building model...")
    model = build_model(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = 0.0
    best_model_path = model_dir / "best_resnet18_gtsrb.pth"

    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "image_size": image_size,
                    "best_val_accuracy": best_val_accuracy,
                    "class_names": None,
                },
                best_model_path,
            )

            print(f"Saved new best model to {best_model_path}")

    total_time = time.time() - start_time

    metrics = {
        "best_val_accuracy": best_val_accuracy,
        "total_training_time_seconds": total_time,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }

    metrics_path = report_dir / "metrics.json"
    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    history_path = report_dir / "training_history.json"
    with open(history_path, "w") as file:
        json.dump(history, file, indent=4)

    plot_path = figure_dir / "training_curves.png"
    plot_training_history(history, plot_path)

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved training history to: {history_path}")
    print(f"Saved training curves to: {plot_path}")


if __name__ == "__main__":
    main()