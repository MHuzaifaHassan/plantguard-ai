
import tensorflow as tf
from tensorflow.keras import layers, models
from datasets import load_dataset
import numpy as np
import os
import matplotlib.pyplot as plt

# =========================
# 1. Load Dataset
# =========================

print("Loading dataset...")

dataset = load_dataset("Project-AgML/tomato_leaf_disease")

print(dataset)

# =========================
# 2. Get Train/Test Data
# =========================

train_data = dataset["train"]

print("Number of images:", len(train_data))

# =========================
# 3. Get Class Names
# =========================

features = train_data.features

label_feature = features["label"]

class_names = label_feature.names

print("Classes:")

for i, name in enumerate(class_names):
    print(i, ":", name)

NUM_CLASSES = len(class_names)

# =========================
# 4. Image Preprocessing
# =========================

IMG_SIZE = 224


def preprocess(example):
    image = example["image"].convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = np.array(image, dtype=np.float32)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, example["label"]


# =========================
# 5. Convert Dataset
# =========================

images = []
labels = []

print("Preparing images...")

for example in train_data:
    image, label = preprocess(example)

    images.append(image)
    labels.append(label)

X = np.array(images)
y = np.array(labels)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 6. Train / Validation Split
# =========================

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))

# =========================
# 7. MobileNetV2
# =========================

base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# =========================
# 8. Classification Head
# =========================

model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.2),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# =========================
# 9. Compile Model
# =========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 10. Train Model
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32
)

# =========================
# 11. Save Model
# =========================

os.makedirs("models", exist_ok=True)

model.save("models/plant_disease_model.keras")

# Save class names
with open("models/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# =========================
# 12. Save Training Graphs
# =========================

os.makedirs("results", exist_ok=True)

# -------------------------
# Accuracy Graph
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -------------------------
# Loss Graph
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================
# 13. Final Message
# =========================

print("\n================================")
print("Training completed!")
print("Model saved successfully!")
print("Training graphs saved successfully!")
print("================================")

print("\nSaved files:")
print("models/plant_disease_model.keras")
print("models/class_names.txt")
print("results/accuracy_curve.png")
print("results/loss_curve.png")

