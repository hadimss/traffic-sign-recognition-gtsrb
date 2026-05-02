"""
Prediction utilities for Traffic Sign Recognition.
"""

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dataset import CLASS_NAMES
from model import build_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_prediction_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_trained_model(
    model_path: str = "models/best_resnet18_gtsrb.pth",
    num_classes: int = 43,
    device: torch.device | None = None,
):
    if device is None:
        device = get_device()

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. "
            "Please train the model first."
        )

    checkpoint = torch.load(model_path, map_location=device)

    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def predict_image(
    image: Image.Image,
    model,
    device: torch.device | None = None,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    if device is None:
        device = get_device()

    image = image.convert("RGB")

    transform = get_prediction_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []

    for probability, class_index in zip(top_probs[0], top_indices[0]):
        class_name = CLASS_NAMES[class_index.item()]
        confidence = probability.item()
        results.append((class_name, confidence))

    return results


if __name__ == "__main__":
    print("Prediction module loaded successfully.")
    print(f"Number of classes: {len(CLASS_NAMES)}")