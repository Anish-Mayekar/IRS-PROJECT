from flask import Flask, request, render_template, jsonify, send_from_directory
import joblib
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import normalize
import cv2
from skimage.feature import local_binary_pattern

app = Flask(__name__)

# Load the model
model_data = joblib.load('image_retrieval_system_with_similarity.joblib')
index = model_data['index']
features = model_data['features']
image_paths = model_data['image_paths']

# Function to preprocess input image and extract features
def extract_color_histogram(image, bins=(8, 8, 8)):
    image = image.convert('RGB')
    image = np.array(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_texture_features(image):
    image = image.convert('L')
    image = np.array(image)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def extract_features(image):
    color_hist = extract_color_histogram(image)
    texture_hist = extract_texture_features(image)
    return np.hstack([color_hist, texture_hist])

def preprocess_image(image):
    # Convert the image to RGB and extract features
    image = image.convert('RGB')
    return extract_features(image)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/retrieve', methods=['POST'])
def retrieve_images():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    uploaded_file = request.files['image']
    img = Image.open(uploaded_file.stream)
    img_features = preprocess_image(img)
    
    # Normalize the features
    img_features = normalize(img_features.reshape(1, -1))
    
    # Perform nearest neighbor search
    distances, indices = index.query(img_features, k=5)  # Get top 5 similar images
    similar_images = [image_paths[i] for i in indices[0]]
    
    return render_template('results.html', similar_images=similar_images)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/jpg/<path:filename>')
def serve_jpg(filename):
    return send_from_directory('jpg', filename)

if __name__ == '__main__':
    app.run(debug=True)
