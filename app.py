import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Use absolute path to ensure model is found
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'flower_model.keras')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Model
print(f"Loading model from {MODEL_PATH}...")
model = None

# Try loading with Keras 3 (standalone) first, then TF Keras (legacy)
try:
    import keras
    print(f"Attempting to load with Keras v{keras.__version__}...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully with Keras 3!")
except Exception as e_keras:
    print(f"Keras 3 load failed: {e_keras}")
    try:
        from tensorflow.keras.models import load_model
        print("Attempting to load with TensorFlow Keras (Legacy)...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully with TF Keras!")
    except Exception as e_tf:
        print(f"Error loading model: {e_tf}")
        # Log error to file
        with open('app_error.log', 'w') as f:
            f.write(f"Keras 3 Error: {e_keras}\nTF Keras Error: {e_tf}")
        model = None

# --- OOD Detection Setup ---
print("Loading ImageNet MobileNetV2 for Pre-check...")
try:
    filter_model = MobileNetV2(weights='imagenet')
    print("Filter model loaded!")
except Exception as e:
    print(f"Failed to load filter model: {e}")
    filter_model = None

# Keywords that indicate an image is likely a flower/plant
FLOWER_KEYWORDS = [
    'flower', 'bloom', 'plant', 'daisy', 'rose', 'tulip', 'sunflower', 
    'dandelion', 'lily', 'orchid', 'garden', 'greenhouse', 'vase', 
    'pot', 'leaf', 'tree', 'vegetable', 'fruit', 'cardoon', 'rapeseed'
]
# ---------------------------

# Class Labels (Must match training order)
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    # Load image with target size 224x224 (MobileNetV2 requirement)
    img = load_img(image_path, target_size=(224, 224))
    
    # Convert to array
    img_array = img_to_array(img)
    
    # Expand dimensions (Batch size = 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize with MobileNetV2 standard: (-1 to 1)
    # This matches the "preprocess_input" from training
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('index'))
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if model is None:
            return render_template('index.html', error="Model not loaded.")
        
        try:
            # Preprocess
            processed_img = preprocess_image(filepath)

            # --- OOD Check ---
            if filter_model:
                try:
                    # MobileNetV2 expects -1 to 1, which processed_img already is
                    filter_preds = filter_model.predict(processed_img)
                    decoded_filter = decode_predictions(filter_preds, top=5)[0]
                    print(f"ImageNet Predictions: {decoded_filter}")

                    is_flower_related = False
                    for _, label, _ in decoded_filter:
                        label_lower = label.lower()
                        if any(k in label_lower for k in FLOWER_KEYWORDS):
                            is_flower_related = True
                            break
                    
                    if not is_flower_related:
                        # It's likely NOT a flower
                        top_detection = decoded_filter[0][1].replace('_', ' ').title()
                        return render_template('index.html', 
                                   error=f"I am not sure this is a flower. It looks like a '{top_detection}'. Please upload a flower image.",
                                   image_url=url_for('static', filename='uploads/' + filename))
                except Exception as e_filter:
                    print(f"Filter check failed: {e_filter}")
                    # Continue to main prediction if filter fails? Or stop?
                    # Let's continue but log it.
                    pass
            # -----------------
            
            # Predict
            predictions = model.predict(processed_img)
            score = tf.nn.softmax(predictions[0])
            
            # Get result
            class_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[class_idx]
            confidence = float(np.max(predictions[0])) * 100
            
            img_url = url_for('static', filename='uploads/' + filename)
            return render_template('index.html', 
                                   prediction=predicted_class, 
                                   confidence=f"{confidence:.2f}", 
                                   image_url=img_url)
                                   
        except Exception as e:
            return render_template('index.html', error=f"Prediction Error: {str(e)}")
            
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
