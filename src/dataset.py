"""
Dataset utilities for Traffic Sign Recognition.

This file creates PyTorch DataLoaders for the GTSRB dataset using
TorchVision. It includes preprocessing, augmentation, and a validation split.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB


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


def get_train_transforms(image_size: int = 224):
    """
    Transformations for training data.

    Includes augmentation to improve generalization.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_eval_transforms(image_size: int = 224):
    """
    Transformations for validation and test data.
    """
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


def create_dataloaders(
    data_dir: str = "data/raw",
    image_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        data_dir: Location of downloaded dataset.
        image_size: Image size used by the model.
        batch_size: Batch size for training.
        val_split: Percentage of training data used for validation.
        num_workers: Number of workers for data loading.
        seed: Random seed for reproducible splitting.

    Returns:
        train_loader, val_loader, test_loader
    """

    data_path = Path(data_dir)

    train_base_dataset = GTSRB(
        root=str(data_path),
        split="train",
        download=False,
        transform=get_train_transforms(image_size),
    )

    val_base_dataset = GTSRB(
        root=str(data_path),
        split="train",
        download=False,
        transform=get_eval_transforms(image_size),
    )

    dataset_size = len(train_base_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)

    test_dataset = GTSRB(
        root=str(data_path),
        split="test",
        download=False,
        transform=get_eval_transforms(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders()

    images, labels = next(iter(train_loader))

    print("DataLoaders created successfully.")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")