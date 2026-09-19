# PlantGuard AI

## Tomato Leaf Disease Detection Using Transfer Learning

PlantGuard AI is a web app that classifies tomato leaf images into different disease and healthy conditions. It uses transfer learning with the MobileNetV2 architecture, and the interface is built with Streamlit.

You give it a tomato leaf image, it processes the image, and it returns the predicted class along with the model's confidence score.

---

## 1. Project Overview

Plant diseases can hurt crop health and reduce yield. Spotting a problem early makes it easier to keep an eye on affected plants and take action.

This project shows how a pretrained convolutional neural network can be adapted for tomato leaf disease classification using transfer learning.

The overall workflow is:

```text
Input Image
     |
     v
Image Preprocessing
     |
     v
MobileNetV2
     |
     v
Transfer Learning
     |
     v
Disease Classification
     |
     v
Prediction and Confidence
     |
     v
Streamlit Web Application
```

---

## 2. Objectives

The main objectives of this project are:

* To build an image classification system for tomato leaves.
* To use a pretrained MobileNetV2 model for transfer learning.
* To freeze the pretrained base layers during initial training.
* To train a custom classification head for tomato leaf conditions.
* To save the trained model for later use.
* To develop a web-based prediction interface using Streamlit.

---

## 3. Dataset

The project uses the Tomato Leaf Disease dataset from Hugging Face.

Dataset:

`Project-AgML/tomato_leaf_disease`

The dataset has around 11,000 images across 10 classes.

### Classes

| ID | Class                                |
| -: | ------------------------------------ |
|  0 | Bacterial Spot                       |
|  1 | Early Blight                         |
|  2 | Healthy                              |
|  3 | Late Blight                          |
|  4 | Leaf Mold                            |
|  5 | Septoria Leaf Spot                   |
|  6 | Spider Mites Two-spotted Spider Mite |
|  7 | Target Spot                          |
|  8 | Tomato Mosaic Virus                  |
|  9 | Tomato Yellow Leaf Curl Virus        |

---

## 4. Model

The project uses MobileNetV2 with ImageNet pretrained weights.

I chose MobileNetV2 as the base because it is a fairly lightweight convolutional neural network that still works well for image classification.

The pretrained convolutional base is frozen during the initial training stage:

```python
base_model.trainable = False
```

A custom classification head is added on top of the pretrained network.

---

## 5. Transfer Learning

Transfer learning lets a model reuse what it learned from a large image dataset on a different classification task.

Here, MobileNetV2 was first trained on ImageNet. Its pretrained feature extraction layers are used as the base of the model.

New layers are then added on top for the 10 tomato leaf classes.

The model structure is:

```text
Input Image
224 x 224 x 3
       |
       v
MobileNetV2
ImageNet Pretrained
       |
       v
Frozen Base Layers
       |
       v
Global Average Pooling
       |
       v
Dropout
       |
       v
Dense Layer
128 Units
       |
       v
Dropout
       |
       v
Softmax Output
10 Classes
```

---

## 6. Image Preprocessing

Each image is converted to RGB and resized to:

```text
224 x 224 pixels
```

After that it goes through the MobileNetV2 preprocessing function:

```python
tf.keras.applications.mobilenet_v2.preprocess_input()
```

---

## 7. Training Configuration

The dataset is split into training and validation sets with an 80/20 split.

The training configuration is:

| Parameter          | Value                           |
| ------------------ | ------------------------------- |
| Base Model         | MobileNetV2                     |
| Pretrained Weights | ImageNet                        |
| Image Size         | 224 x 224                       |
| Training Split     | 80%                             |
| Validation Split   | 20%                             |
| Optimizer          | Adam                            |
| Loss Function      | Sparse Categorical Crossentropy |
| Batch Size         | 32                              |
| Epochs             | 5                               |
| Number of Classes  | 10                              |

The pretrained base layers stay frozen during this initial training.

---

## 8. Model Saving

After training, the full model is saved in Keras format:

```text
models/plant_disease_model.keras
```

The class names are saved separately:

```text
models/class_names.txt
```

The saved model can be loaded later without training again:

```python
model = tf.keras.models.load_model(
    "models/plant_disease_model.keras"
)
```

---

## 9. Prediction

The `predict.py` script loads the trained model and runs it on a tomato leaf image.

The prediction process is:

```text
Test Image
     |
     v
Resize to 224 x 224
     |
     v
MobileNetV2 Preprocessing
     |
     v
Trained Model
     |
     v
Class Probabilities
     |
     v
Predicted Class
     |
     v
Confidence Score
```

For example, a test image may give a result like this:

```text
Disease: Late Blight
Confidence: 79.03%
```

The confidence value is the model's probability for the predicted class. It is not a guaranteed diagnosis.

---

## 10. Streamlit Application

The project includes a web interface built with Streamlit.

With the app, users can:

1. Upload a tomato leaf image.
2. Preview the uploaded image.
3. Run the trained model.
4. See the predicted class.
5. See the prediction confidence.
6. Read more information about the prediction.

Start the app with:

```powershell
streamlit run app.py
```

---

## 11. Project Structure

```text
plant_disease_detector/
|
├── dataset/
|
├── models/
|   ├── plant_disease_model.keras
|   └── class_names.txt
|
├── test_images/
|   └── tomato.jpg
|
├── train.py
├── predict.py
├── evaluate.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 12. Installation

Create a Python virtual environment:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
venv\Scripts\activate
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

---

## 13. Running the Project

### Train the model

```powershell
python train.py
```

### Test a single image

Put an image inside the `test_images` folder and run:

```powershell
python predict.py
```

### Evaluate the model

```powershell
python evaluate.py
```

### Start the web application

```powershell
streamlit run app.py
```

---

## 14. Requirements

The project uses these main Python libraries:

```text
tensorflow
streamlit
numpy
pillow
matplotlib
scikit-learn
datasets
```

Install them with:

```powershell
pip install -r requirements.txt
```

---

## 15. Model Evaluation

After training, the model was evaluated on the 20% validation split (2,200 images, 220 per class) using `evaluate.py`.

| Metric              | Result |
| ------------------- | ------ |
| Validation Accuracy | 89.18% |
| Precision           | 89.21% |
| Recall              | 89.18% |
| F1-Score            | 89.16% |

Precision, recall and F1-score are weighted averages. The validation set has the same number of images for every class, so the macro average gives about the same result.

### Per-Class Results

| Class                                | Precision | Recall | F1-Score | Support |
| ------------------------------------ | --------: | -----: | -------: | ------: |
| Bacterial Spot                       |      0.92 |   0.91 |     0.92 |     220 |
| Early Blight                         |      0.81 |   0.81 |     0.81 |     220 |
| Healthy                              |      0.91 |   0.97 |     0.94 |     220 |
| Late Blight                          |      0.92 |   0.86 |     0.89 |     220 |
| Leaf Mold                            |      0.89 |   0.94 |     0.92 |     220 |
| Septoria Leaf Spot                   |      0.87 |   0.85 |     0.86 |     220 |
| Spider Mites Two-spotted Spider Mite |      0.85 |   0.88 |     0.86 |     220 |
| Target Spot                          |      0.81 |   0.78 |     0.79 |     220 |
| Tomato Mosaic Virus                  |      0.99 |   0.96 |     0.97 |     220 |
| Tomato Yellow Leaf Curl Virus        |      0.95 |   0.95 |     0.95 |     220 |

### Observations

* Tomato Mosaic Virus (F1 0.97) and Tomato Yellow Leaf Curl Virus (F1 0.95) were the easiest classes for the model.
* Target Spot (F1 0.79) and Early Blight (F1 0.81) were the hardest.
* Target Spot images were most often mixed up with Spider Mites (19 images) and Healthy (13 images).
* Early Blight was mainly confused with Late Blight and Septoria Leaf Spot (10 images each).
* Late Blight was mostly confused with Early Blight and Leaf Mold (11 images each).

These results come from the validation split only, after 5 epochs with the base layers frozen. They show how the model performs on this dataset and may not carry over to photos taken in real field conditions.

---

## 16. Future Improvements

Some ways the project could be extended:

* Fine-tuning selected MobileNetV2 layers.
* Applying data augmentation.
* Increasing dataset diversity.
* Comparing MobileNetV2 with other architectures such as ResNet.
* Adding prediction history.
* Adding more plant species.
* Improving the deployment setup.
* Deploying the application to a cloud platform.

---

## 17. Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* Hugging Face Datasets
* NumPy
* Scikit-learn
* Pillow
* Streamlit

---

## 18. Project Status

The project currently includes:

* A tomato leaf image dataset.
* A MobileNetV2 transfer learning model.
* Frozen pretrained base layers.
* A trained Keras model.
* A prediction script.
* An evaluation script.
* A Streamlit web application.

Right now it works as a prototype for tomato leaf disease classification.

---

## 19. Disclaimer

This project is for educational and demonstration purposes. The model's predictions may not always be right and should not be treated as a confirmed agricultural diagnosis. For crop disease management decisions, please consult appropriate agricultural expertise.