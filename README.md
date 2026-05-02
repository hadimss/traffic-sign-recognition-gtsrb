# Traffic Sign Recognition using Fine-Tuned ResNet-18

## Overview
This project focuses on traffic sign recognition using deep learning. The goal is to classify cropped traffic sign images into their correct categories using a fine-tuned pretrained ResNet-18 model.

The project uses the GTSRB dataset, a widely used benchmark dataset for traffic sign classification.

## Problem Statement
Traffic sign recognition is an important component of driver-assistance and autonomous driving systems. The system should take an image of a traffic sign as input and predict the correct class, such as speed limit, stop, yield, or no entry.

## Project Objectives
- Preprocess traffic sign image data.
- Fine-tune a pretrained ResNet-18 model.
- Train and validate the model on the GTSRB dataset.
- Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrix.
- Build a simple Streamlit app for traffic sign prediction.

## Dataset
The project uses the German Traffic Sign Recognition Benchmark, also known as GTSRB.

The dataset contains traffic sign images from 43 classes.

## Methodology
1. Data loading and exploration
2. Image preprocessing and augmentation
3. Model fine-tuning
4. Training and validation
5. Model evaluation
6. Streamlit application for prediction

## Tech Stack
- Python
- PyTorch
- TorchVision
- Hugging Face Datasets
- scikit-learn
- Matplotlib
- Streamlit

## Project Structure

```text
traffic-sign-recognition-gtsrb/
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   └── config.yaml
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── LICENSE

## Current Status
Project setup in progress.

## Future Work
- Add real-time webcam prediction.
- Compare ResNet-18 with MobileNetV3.
- Deploy the Streamlit app online.
- Extend the project from classification to traffic sign detection.
