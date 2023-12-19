import os
import numpy as np
from skimage import io, transform
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load and resize images
def load_and_resize_images(folder, target_size=(100, 100)):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = io.imread(img_path)
                img_resized = transform.resize(img, target_size)
                images.append(img_resized.flatten())  # Flatten the resized image
                labels.append(1 if subfolder.lower() == 'malignant' else 0)
    return np.array(images), np.array(labels)

# Load and resize images from the TRAIN folder
train_folder = '/Users/shahadsaeed/Desktop/melanoma_cancer_dataset/train'
X_train, y_train = load_and_resize_images(train_folder)

# Load and resize images from the TEST folder
test_folder = '/Users/shahadsaeed/Desktop/melanoma_cancer_dataset/test'
X_test, y_test = load_and_resize_images(test_folder)

# Perform PCA for dimensionality reduction
num_components = 100  # Adjust this based on your needs
pca = PCA(n_components=num_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Support Vector Machine (SVM) classifier
print("Training SVM classifier...")
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train_pca, y_train)
print("SVM training complete.")

# Decision Tree classifier
print("Training Decision Tree classifier...")
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_pca, y_train)
print("Decision Tree training complete.")

# Random Forest classifier
print("Training Random Forest classifier...")
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_pca, y_train)
print("Random Forest training complete.")

# Naive Bayes classifier
print("Training Naive Bayes classifier...")
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_pca, y_train)
print("Naive Bayes training complete.")

# Evaluation on the TEST set
print("Evaluating classifiers on the TEST set...")

# Support Vector Machine (SVM) predictions
print("Making predictions with SVM...")
svm_predictions = svm_classifier.predict(X_test_pca)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_classification_report = classification_report(y_test, svm_predictions)
print("SVM Test Accuracy:", svm_accuracy)
print("SVM Test Classification Report:\n", svm_classification_report)

# Decision Tree predictions
print("Making predictions with Decision Tree...")
dt_predictions = dt_classifier.predict(X_test_pca)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_classification_report = classification_report(y_test, dt_predictions)
print("Decision Tree Test Accuracy:", dt_accuracy)
print("Decision Tree Test Classification Report:\n", dt_classification_report)

# Random Forest predictions
print("Making predictions with Random Forest...")
rf_predictions = rf_classifier.predict(X_test_pca)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_classification_report = classification_report(y_test, rf_predictions)
print("Random Forest Test Accuracy:", rf_accuracy)
print("Random Forest Test Classification Report:\n", rf_classification_report)

# Naive Bayes predictions
print("Making predictions with Naive Bayes...")
nb_predictions = nb_classifier.predict(X_test_pca)
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_classification_report = classification_report(y_test, nb_predictions)
print("Naive Bayes Test Accuracy:", nb_accuracy)
print("Naive Bayes Test Classification Report:\n", nb_classification_report)

# Confusion matrix visualization
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Plot confusion matrices for each classifier
plot_confusion_matrix(y_test, svm_predictions, "SVM Confusion Matrix")
plot_confusion_matrix(y_test, dt_predictions, "Decision Tree Confusion Matrix")
plot_confusion_matrix(y_test, rf_predictions, "Random Forest Confusion Matrix")
plot_confusion_matrix(y_test, nb_predictions, "Naive Bayes Confusion Matrix")

# Print and compare metrics for the test set
print("\nTest Set Metrics Comparison:")
print("SVM Test Accuracy:", svm_accuracy)
print("Decision Tree Test Accuracy:", dt_accuracy)
print("Random Forest Test Accuracy:", rf_accuracy)
print("Naive Bayes Test Accuracy:", nb_accuracy)

# Accuracy Comparison Bar Plot
classifiers = ['SVM', 'Decision Tree', 'Random Forest', 'Naive Bayes']
accuracies = [svm_accuracy, dt_accuracy, rf_accuracy, nb_accuracy]

plt.figure(figsize=(8, 6))
sns.barplot(x=classifiers, y=accuracies, palette="viridis")
plt.title('Test Set Accuracy Comparison')
plt.ylim(0, 1.0)
plt.ylabel('Accuracy')
plt.show()
