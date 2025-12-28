"""
Configuration file for Face Recognition Dataset Project

Project Information:
- Project ID: 22
- Title: Face Recognition Dataset
- Category: Image Data
- Description: Facial recognition dataset with labeled face images across multiple identities
- Technologies: PNG, JPG, NumPy, OpenCV, Face Recognition
- Difficulty: Intermediate

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

import os

# Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Image settings
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
IMAGE_SIZE = (160, 160)  # Standard face recognition image size
IMAGE_CHANNELS = 3  # RGB

# Face recognition settings
FACE_DETECTION_MODEL = 'hog'  # 'hog' or 'cnn'
NUM_JITTERS = 1  # Number of times to re-sample the face when calculating encoding
TOLERANCE = 0.6  # Lower is more strict (0.6 is typical)

# Training settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Output settings
SAVE_MODEL = True
MODEL_NAME = 'face_recognition_model.pkl'

