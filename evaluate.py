
import tensorflow as tf
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

IMG_SIZE = 224

print("Loading dataset...")
dataset = load_dataset("Project-AgML/tomato_leaf_disease")
data = dataset["train"]

class_names = data.features["label"].names

images = []
labels = []

print("Preparing images...")

for example in data:
    image = example["image"].convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = np.array(image, dtype=np.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    images.append(image)
    labels.append(example["label"])

X = np.array(images)
y = np.array(labels)

_, X_val, _, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Loading trained model...")

model = tf.keras.models.load_model(
    "models/plant_disease_model.keras"
)

print("Generating predictions...")

predictions = model.predict(
    X_val,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_val, y_pred)

precision = precision_score(
    y_val,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_val,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n==============================")
print("MODEL EVALUATION RESULTS")
print("==============================")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

# Create confusion matrix
cm = confusion_matrix(y_val, y_pred)

print("\nCONFUSION MATRIX")
print("==============================")
print(cm)

# Save confusion matrix as PNG
plt.figure(figsize=(12, 10))

plt.imshow(cm, interpolation="nearest")
plt.title("Plant Disease Detection - Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))

plt.xticks(
    tick_marks,
    class_names,
    rotation=90
)

plt.yticks(
    tick_marks,
    class_names
)

# Add numbers inside the matrix
threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved successfully!")
print("File: confusion_matrix.png")

print("\nEvaluation completed successfully.")
