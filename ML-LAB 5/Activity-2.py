import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

DATASET_FOLDER = "brain_tumor_dataset"
IMG_SIZE = 64

folder = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(folder, DATASET_FOLDER)

# Step 1: Load the dataset and preprocess all MRI images
# (grayscale, resize to 64x64, flatten)
images = []
labels = []
filenames = []

for label_name, label_value in [("no", 0), ("yes", 1)]:
    class_folder = os.path.join(dataset_path, label_name)
    for fname in sorted(os.listdir(class_folder)):
        fpath = os.path.join(class_folder, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipped unreadable file: {fpath}")
            continue
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img_resized.flatten())
        labels.append(label_value)
        filenames.append(fpath)

X = np.array(images)
y = np.array(labels)

print(f"Loaded {len(X)} images")
print(f"  No tumor (0): {np.sum(y == 0)}")
print(f"  Tumor   (1): {np.sum(y == 1)}")
print(f"Flattened feature size: {X.shape[1]} (= {IMG_SIZE} x {IMG_SIZE})")

# Step: Split into training and testing sets (80:20)
X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    X, y, filenames, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} images   Test: {X_test.shape[0]} images")

# Step: Normalize features using StandardScaler()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step: Train an SVM (RBF kernel) with C=10 and gamma='scale'
svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(X_train_scaled, y_train)

# Step: Evaluate performance using accuracy, classification report, confusion matrix
y_pred = svm.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Tumor", "Tumor"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=["No Tumor", "Tumor"]).plot(
    cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - SVM (RBF, C=10)")
plt.tight_layout()
plt.savefig(os.path.join(folder, "svm_confusion_matrix.png"), dpi=150)
plt.show()

# Step: Display a few test images with predicted and actual labels
label_text = {0: "No Tumor", 1: "Tumor"}
n_show = min(8, len(X_test))
rng = np.random.RandomState(42)
show_idx = rng.choice(len(X_test), size=n_show, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(13, 7))
for ax, idx in zip(axes.flat, show_idx):
    img = cv2.imread(files_test[idx], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    ax.imshow(img, cmap="gray")
    actual = label_text[y_test[idx]]
    predicted = label_text[y_pred[idx]]
    color = "green" if actual == predicted else "red"
    ax.set_title(f"Actual: {actual}\nPredicted: {predicted}", color=color, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(folder, "svm_sample_predictions.png"), dpi=150)
plt.show()