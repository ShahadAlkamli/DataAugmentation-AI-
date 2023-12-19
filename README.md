# Melanoma Skin Cancer Classification

## Overview

This repository contains code for the classification of melanoma skin cancer images using various machine learning classifiers. The dataset used for this project was obtained from Kaggle and consists of two folders: `TRAIN` (9600 images) and `TEST` (1000 images). Each folder includes subfolders for malignant and benign images.

## Code Structure

### 1. Image Loading and Preprocessing
- Images are loaded and resized using the `load_and_resize_images` function.
- Principal Component Analysis (PCA) is applied for dimensionality reduction.

### 2. Classifiers
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Naive Bayes

### 3. Training and Evaluation
- Classifiers are trained using the preprocessed data.
- The `melanoma_classification.py` script evaluates classifiers on the `TEST` set using metrics such as accuracy, classification report, and confusion matrices.
- Confusion matrices are visualized using the `plot_confusion_matrix` function.

### 4. Data Augmentation
- Data augmentation is performed on the `TRAIN` set using the `ImageDataGenerator` from TensorFlow.
- The `augmentation.py` script is responsible for data augmentation.

### 5. Augmented Data Evaluation
- The augmented data is reevaluated using the same classifiers for comparison.


