import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Load Model
# =========================

model = tf.keras.models.load_model(
    "models/plant_disease_model.keras"
)

# =========================
# Load Class Names
# =========================

with open("models/class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# =========================
# Load Test Image
# =========================

image_path = "test_images/tomato.jpg"

image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))

# Convert to numpy
image_array = np.array(image, dtype=np.float32)

# MobileNetV2 preprocessing
image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    image_array
)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# =========================
# Prediction
# =========================

predictions = model.predict(image_array)

predicted_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_index] * 100

predicted_class = class_names[predicted_index]

# =========================
# Result
# =========================

print("\n==============================")
print("🌱 Plant Disease Prediction")
print("==============================")

print("Disease:", predicted_class)
print("Confidence: {:.2f}%".format(confidence))

print("==============================")