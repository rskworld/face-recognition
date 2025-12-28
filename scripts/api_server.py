"""
REST API Server for Face Recognition

This module provides a Flask-based REST API for face recognition services.

Project Information:
- Project ID: 22
- Title: Face Recognition Dataset
- Category: Image Data
- Technologies: PNG, JPG, NumPy, OpenCV, Face Recognition

Contact Information:
RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277
Address: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in/
Year: 2026
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import base64
from scripts.recognize_faces import FaceRecognizer
import config
import os

# Try to import advanced features (optional)
try:
    from scripts.advanced_features import FaceVerifier, FaceClustering, FaceQualityAssessment
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    print("Warning: Advanced features not available. Some API endpoints may not work.")

app = Flask(__name__)

# Initialize recognizer
recognizer = None
verifier = None
quality_assessor = None

if ADVANCED_FEATURES_AVAILABLE:
    verifier = FaceVerifier()
    quality_assessor = FaceQualityAssessment()


def init_recognizer():
    """Initialize the face recognizer."""
    global recognizer
    recognizer = FaceRecognizer()
    if os.path.exists(recognizer.model_path):
        recognizer.load_model()
    else:
        recognizer.train(config.TRAIN_DIR, save_model=True)


def decode_image(image_data: str) -> np.ndarray:
    """
    Decode base64 image data.
    
    Args:
        image_data: Base64 encoded image string
        
    Returns:
        Decoded image as numpy array
    """
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Face Recognition API',
        'version': '1.0.0'
    })


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """
    Recognize faces in an image.
    
    Request body:
    {
        "image": "base64_encoded_image"
    }
    """
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image = decode_image(data['image'])
        # PIL images are already in RGB format, no conversion needed
        rgb_image = image
        
        if recognizer is None:
            init_recognizer()
        
        results = recognizer.recognize_from_array(rgb_image)
        
        return jsonify({
            'success': True,
            'faces_detected': len(results),
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/verify', methods=['POST'])
def verify():
    """
    Verify if two images contain the same person.
    
    Request body:
    {
        "image1": "base64_encoded_image1",
        "image2": "base64_encoded_image2"
    }
    """
    if not ADVANCED_FEATURES_AVAILABLE or verifier is None:
        return jsonify({'error': 'Face verification feature not available'}), 503
    
    try:
        data = request.json
        if 'image1' not in data or 'image2' not in data:
            return jsonify({'error': 'Both images required'}), 400
        
        image1 = decode_image(data['image1'])
        image2 = decode_image(data['image2'])
        
        # PIL images are already in RGB format, no conversion needed
        rgb1 = image1
        rgb2 = image2
        
        from scripts.preprocess import get_face_encoding
        
        encoding1 = get_face_encoding(rgb1)
        encoding2 = get_face_encoding(rgb2)
        
        if encoding1 is None or encoding2 is None:
            return jsonify({'error': 'Could not detect face in one or both images'}), 400
        
        result = verifier.verify(encoding1, encoding2)
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/quality', methods=['POST'])
def assess_quality():
    """
    Assess quality of face image.
    
    Request body:
    {
        "image": "base64_encoded_image"
    }
    """
    if not ADVANCED_FEATURES_AVAILABLE or quality_assessor is None:
        return jsonify({'error': 'Quality assessment feature not available'}), 503
    
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image = decode_image(data['image'])
        # PIL images are already in RGB format, no conversion needed
        rgb_image = image
        
        from scripts.preprocess import detect_faces
        
        face_locations = detect_faces(rgb_image)
        
        if len(face_locations) == 0:
            return jsonify({'error': 'No face detected'}), 400
        
        quality = quality_assessor.assess(rgb_image, face_locations[0])
        
        return jsonify({
            'success': True,
            'quality': quality
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster', methods=['POST'])
def cluster():
    """
    Cluster multiple face encodings.
    
    Request body:
    {
        "encodings": [[...], [...], ...]
    }
    """
    if not ADVANCED_FEATURES_AVAILABLE:
        return jsonify({'error': 'Face clustering feature not available'}), 503
    
    try:
        data = request.json
        if 'encodings' not in data:
            return jsonify({'error': 'No encodings provided'}), 400
        
        encodings = [np.array(e) for e in data['encodings']]
        
        clusterer = FaceClustering()
        result = clusterer.cluster(encodings)
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Face Recognition API Server")
    print("=" * 60)
    print(f"Project: Face Recognition Dataset (ID: 22)")
    print(f"RSK World - https://rskworld.in/")
    print("=" * 60)
    print("\nInitializing recognizer...")
    init_recognizer()
    print("\nStarting server on http://localhost:5000")
    print("API endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/recognize - Recognize faces")
    print("  POST /api/verify - Verify two faces")
    print("  POST /api/quality - Assess image quality")
    print("  POST /api/cluster - Cluster face encodings")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

