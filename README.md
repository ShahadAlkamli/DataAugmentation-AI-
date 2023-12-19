# Cancer Classification Project

This repository contains code for a Cancer Classification Project, focusing on melanoma skin cancer images.

# Cancer Classification Project

This repository contains code for a Cancer Classification Project, focusing on melanoma skin cancer images.

## Project Overview

Skin cancer is a critical health concern, and early detection is vital for effective treatment. This project leverages machine learning techniques to classify melanoma skin cancer images. The dataset used for this project was obtained from Kaggle, consisting of two main folders: `TRAIN` (9600 images) and `TEST` (1000 images). Each folder includes subfolders for malignant and benign cases.
**[Link to the Dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)**


## Folder Structure

- `code/`: Contains the project's main scripts.
- `augmented_images/`: Stores augmented images generated during data augmentation.
- `results_and_analysis/`: Holds results and analysis of experiments.


## How to Run

1. **Data Augmentation:**
    - Navigate to the `code/` directory.
    - Run `Augmentation.py` to perform data augmentation on the original images.

2. **Merging Original and Augmented Data:**
    - Run `Merge.py` to merge the original and augmented datasets for training.

3. **Classifier Evaluation:**
    - Run `Evaluation.py` to train and evaluate classifiers on the merged dataset.


## Reproducing Experiments

1. Clone this repository:
    ```bash
    git clone https://github.com/ShahadAlkamli/IT504.git
    cd IT504
    ```

2. Install dependencies:
    - Python (3.x)
    - scikit-learn
    - TensorFlow (for data augmentation)
    - matplotlib
    - seaborn
    - numpy
    - scikit-image

    You can install them using:
    ```bash
    pip install scikit-learn tensorflow matplotlib seaborn numpy scikit-image
    ```

3. Run experiments:
    - Follow the steps under "How to Run."



