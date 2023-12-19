# Cancer Classification Project

This repository contains code for a Cancer Classification Project, focusing on melanoma skin cancer images.

## How to Run

1. **Data Augmentation:**
    - Navigate to the `code/` directory.
    - Run `Augmentation.py` to perform data augmentation on the original images.

2. **Merging Original and Augmented Data:**
    - Run `Merge.py` to merge the original and augmented datasets for training.

3. **Classifier Evaluation:**
    - Run `Evaluation.py` to train and evaluate classifiers on the merged dataset.

## Folder Structure

- `code/`: Contains the project's main scripts.
- `augmented_images/`: Stores augmented images generated during data augmentation.
- `results_and_analysis/`: Holds results and analysis of experiments.

## Reproducing Experiments

1. Clone this repository:
    ```bash
    git clone https://github.com/SahhadAlkamli/IT504.git
    cd IT504
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run experiments:
    - Follow the steps under "How to Run."

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
