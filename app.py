from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Set up CORS
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Path to your "my_trained_cnn.tflite" file
TFLITE_MODEL_PATH = 'my_trained_cnn.tflite'

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_data):
    # Load the image file from bytes data
    image = Image.open(io.BytesIO(image_data))

    # Resize the image and normalize pixel values
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    
    # Convert the image to float32 data type
    image_array = image_array.astype(np.float32)
    
    # Add a batch dimension and return
    return np.expand_dims(image_array, axis=0)

@app.route("/")
def home():
    return jsonify({
        "message": "Server is running"
    })

@app.route("/predict/", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        image_data = file.read()
        preprocessed_image = preprocess_image(image_data)
        
        # Set the tensor to point to the input data to be inferred
        interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
        
        # Run the inference
        interpreter.invoke()
        
        # Retrieve the prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get the predicted class
        predicted_class = np.argmax(output_data[0])
        
        # Map the predicted class to the class name
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        predicted_class_name = class_names[predicted_class]
        
        return jsonify({"predicted_class": predicted_class_name})

if __name__ == "__main__":
    app.run(port=8000, debug=True)
