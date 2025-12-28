"""
Face Recognition Script

This script performs face recognition on images using the trained model.

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

import numpy as np
import cv2
import face_recognition
import pickle
import os
from typing import List, Tuple, Optional, Dict
import config
from scripts.preprocess import get_face_encoding


class FaceRecognizer:
    """
    Face recognition system that can train on a dataset and recognize faces.
    """
    
    def __init__(self, tolerance: float = None):
        """
        Initialize the face recognizer.
        
        Args:
            tolerance: How much distance between faces to consider it a match (default from config)
        """
        self.tolerance = tolerance or config.TOLERANCE
        self.known_face_encodings = []
        self.known_face_names = []
        self.model_path = os.path.join(config.MODELS_DIR, config.MODEL_NAME)
        
    def train(self, data_dir: str, save_model: bool = True):
        """
        Train the recognizer on a dataset.
        
        Args:
            data_dir: Directory containing person subdirectories
            save_model: Whether to save the trained model
        """
        from scripts.load_dataset import FaceDatasetLoader
        
        loader = FaceDatasetLoader(data_dir)
        images, labels, label_mapping = loader.load()
        
        print(f"Training on {len(images)} images from {len(label_mapping)} identities...")
        
        self.known_face_encodings = []
        self.known_face_names = []
        
        for i, image in enumerate(images):
            # Get face encoding
            encoding = get_face_encoding(image)
            
            if encoding is not None:
                person_name = label_mapping[labels[i]]
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(person_name)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(images)} images")
        
        print(f"Trained on {len(self.known_face_encodings)} face encodings")
        
        if save_model:
            self.save_model()
    
    def recognize(self, image_path: str) -> List[Dict]:
        """
        Recognize faces in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of recognition results with name and confidence
        """
        if len(self.known_face_encodings) == 0:
            if os.path.exists(self.model_path):
                self.load_model()
            else:
                raise ValueError("No trained model found. Please train the model first.")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_image, 
            model=config.FACE_DETECTION_MODEL
        )
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(
            rgb_image, 
            face_locations,
            num_jitters=config.NUM_JITTERS
        )
        
        results = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings,
                face_encoding,
                tolerance=self.tolerance
            )
            
            # Calculate distances
            face_distances = face_recognition.face_distance(
                self.known_face_encodings,
                face_encoding
            )
            
            # Find best match
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                distance = face_distances[best_match_index]
                confidence = 1 - distance  # Convert distance to confidence
            else:
                name = "Unknown"
                confidence = 0.0
            
            results.append({
                'name': name,
                'confidence': confidence,
                'location': face_location,
                'distance': face_distances[best_match_index] if matches[best_match_index] else None
            })
        
        return results
    
    def recognize_from_array(self, image: np.ndarray) -> List[Dict]:
        """
        Recognize faces from a numpy array image.
        
        Args:
            image: Image as numpy array (RGB format)
            
        Returns:
            List of recognition results
        """
        if len(self.known_face_encodings) == 0:
            if os.path.exists(self.model_path):
                self.load_model()
            else:
                raise ValueError("No trained model found. Please train the model first.")
        
        # Detect faces
        face_locations = face_recognition.face_locations(
            image, 
            model=config.FACE_DETECTION_MODEL
        )
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(
            image, 
            face_locations,
            num_jitters=config.NUM_JITTERS
        )
        
        results = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                self.known_face_encodings,
                face_encoding,
                tolerance=self.tolerance
            )
            
            face_distances = face_recognition.face_distance(
                self.known_face_encodings,
                face_encoding
            )
            
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                distance = face_distances[best_match_index]
                confidence = 1 - distance
            else:
                name = "Unknown"
                confidence = 0.0
            
            results.append({
                'name': name,
                'confidence': confidence,
                'location': face_location,
                'distance': face_distances[best_match_index] if matches[best_match_index] else None
            })
        
        return results
    
    def save_model(self):
        """Save the trained model to disk."""
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        
        model_data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names,
            'tolerance': self.tolerance
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.known_face_encodings = model_data['encodings']
        self.known_face_names = model_data['names']
        self.tolerance = model_data.get('tolerance', config.TOLERANCE)
        
        print(f"Model loaded from {self.model_path}")
        print(f"Loaded {len(self.known_face_encodings)} face encodings")


if __name__ == "__main__":
    # Example usage
    recognizer = FaceRecognizer()
    
    # Train the model
    print("Training face recognizer...")
    recognizer.train(config.TRAIN_DIR, save_model=True)
    
    # Example: Recognize faces in an image
    # results = recognizer.recognize('path/to/test/image.jpg')
    # for result in results:
    #     print(f"Name: {result['name']}, Confidence: {result['confidence']:.2f}")

