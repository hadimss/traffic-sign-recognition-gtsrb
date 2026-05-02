"""
Model definition for Traffic Sign Recognition.

This file builds a pretrained ResNet-18 model and replaces the final
classification layer so it can predict the 43 GTSRB traffic sign classes.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes: int = 43, freeze_backbone: bool = False) -> nn.Module:
    """
    Build a pretrained ResNet-18 model for traffic sign classification.

    Args:
        num_classes (int): Number of output classes.
        freeze_backbone (bool): If True, freezes feature extractor layers.

    Returns:
        nn.Module: ResNet-18 model adapted for GTSRB classification.
    """

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = build_model(num_classes=43)
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)

    print("Model created successfully.")
    print(f"Output shape: {output.shape}")