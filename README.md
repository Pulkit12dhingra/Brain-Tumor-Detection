# Brain Tumor CNN Classifier

This repository contains a Convolutional Neural Network (CNN) based classifier for detecting brain tumors from MRI images.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)

## Introduction
This project aims to classify brain MRI images into tumor and non-tumor categories using a deep learning approach. The model is built using Convolutional Neural Networks (CNNs) to achieve high accuracy in classification.

## Dataset
The dataset used for training and testing the model consists of labeled MRI images. Ensure you have the dataset in the appropriate directory before running the model.
Dataset Link:-  https://drive.google.com/drive/folders/1rKyf_-UVWg5G4mEnw_5_faDJrXjnu5H0?usp=share_link
## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/brain_tumor_CNN_classifier.git
cd brain_tumor_CNN_classifier
pip install -r requirements.txt
```

## Usage
To train the model, run the Jupyter notebook:

```bash
jupyter notebook Brain_Tumor_CNN_Modeling.ipynb
```

To use the TKINTER app for image prediction, run:

```bash
python APP.py
```

## Model Architecture
The CNN model consists of multiple convolutional layers followed by max-pooling layers, dropout layers, and fully connected layers. The architecture is designed to extract features from MRI images and classify them accurately.

