# Flower Classification App 🌸

This project is a Flask-based web application for classifying flower species using a Deep Learning model (MobileNetV2) trained on the `tf_flowers` dataset.

## Features
- 🌸 Classifies 5 types of flowers: Daisy, Dandelion, Roses, Sunflowers, Tulips
- 🖼️ Preview uploaded images before prediction
- ⚡ Fast inference using MobileNetV2 Transfer Learning
- 📈 Displays prediction confidence score with animated progress bar
- 🛡️ **Out-of-Distribution Detection**: Filters non-flower images (humans, cars, etc.)
- 🎨 Modern glassmorphism UI design
- ⚠️ Clear disclaimers about model limitations
- 🔄 Easy "Try Again" functionality
- 📱 Responsive design for all devices

## Project Structure
```
flower-classification-app/
│
├── model/
│   └── flower_model.keras  <-- Place your trained model here
├── static/
│   └── uploads/            <-- Temporarily saves uploaded images
├── templates/
│   └── index.html          <-- Frontend UI
├── app.py                  <-- Flask Application Logic
├── requirements.txt        <-- Python Dependencies
└── README.md               <-- Documentation
```

## Setup & Installation

1.  **Clone/Download the project.**
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Place the Model:**
    - Download your trained model from Google Drive.
    - Rename it to `flower_model.keras`.
    - Move it to the `model/` directory.

## How to Run
Run the Flask app:
```bash
python app.py
```
Open your browser and verify the local url, usually: `http://127.0.0.1:5000/`

## Model Details
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Input Size:** 224x224 pixels
- **Training Framework:** TensorFlow / Keras

## Documentation
- **Dataset:** [TensorFlow Flowers Dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
- **Training Notebook:** [Google Colab Link](https://colab.research.google.com/) *(Upload `flower_training.ipynb` to run)*
- **GitHub Repository:** [Optional - Insert Link Here]

## Author
Name: Thifal Hasna Aulia Habibah
NIM: 301220020

