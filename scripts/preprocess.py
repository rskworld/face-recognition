"""
Image Preprocessing for Face Recognition Dataset

This script preprocesses face images for training and recognition.

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
from typing import List, Tuple, Optional
import config


def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image.
    
    Args:
        image: Input image (RGB format)
        
    Returns:
        List of face locations (top, right, bottom, left)
    """
    face_locations = face_recognition.face_locations(
        image, 
        model=config.FACE_DETECTION_MODEL
    )
    return face_locations


def extract_face(image: np.ndarray, face_location: Tuple[int, int, int, int], 
                 size: Tuple[int, int] = None) -> Optional[np.ndarray]:
    """
    Extract and resize a face from an image.
    
    Args:
        image: Input image (RGB format)
        face_location: Face location (top, right, bottom, left)
        size: Target size (width, height), default from config
        
    Returns:
        Extracted and resized face image, or None if extraction fails
    """
    if size is None:
        size = config.IMAGE_SIZE
    
    top, right, bottom, left = face_location
    
    # Extract face region
    face_image = image[top:bottom, left:right]
    
    if face_image.size == 0:
        return None
    
    # Resize to target size
    face_image = cv2.resize(face_image, size)
    
    return face_image


def preprocess_image(image: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Preprocess a single image.
    
    Args:
        image: Input image (RGB format)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image
    """
    # Convert to float
    processed = image.astype(np.float32)
    
    # Normalize to [0, 1] if requested
    if normalize:
        processed = processed / 255.0
    
    return processed


def preprocess_faces(data_dir: str, output_dir: str = None) -> List[np.ndarray]:
    """
    Preprocess all face images in a directory.
    
    Args:
        data_dir: Directory containing person subdirectories
        output_dir: Optional directory to save preprocessed images
        
    Returns:
        List of preprocessed face images
    """
    import os
    from scripts.load_dataset import FaceDatasetLoader
    
    loader = FaceDatasetLoader(data_dir)
    images, labels, label_mapping = loader.load()
    
    processed_images = []
    processed_labels = []
    
    print(f"Preprocessing {len(images)} images...")
    
    for i, image in enumerate(images):
        # Detect faces
        face_locations = detect_faces(image)
        
        if len(face_locations) > 0:
            # Use the first detected face
            face_location = face_locations[0]
            face_image = extract_face(image, face_location)
            
            if face_image is not None:
                # Preprocess the face
                processed_face = preprocess_image(face_image, normalize=True)
                processed_images.append(processed_face)
                processed_labels.append(labels[i])
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(images)} images")
    
    print(f"Successfully processed {len(processed_images)} faces")
    
    return np.array(processed_images), np.array(processed_labels)


def get_face_encoding(image: np.ndarray, num_jitters: int = None) -> Optional[np.ndarray]:
    """
    Get face encoding for recognition.
    
    Args:
        image: Input image (RGB format)
        num_jitters: Number of times to re-sample the face (default from config)
        
    Returns:
        Face encoding (128-dimensional vector) or None if no face detected
    """
    if num_jitters is None:
        num_jitters = config.NUM_JITTERS
    
    # Detect face
    face_locations = detect_faces(image)
    
    if len(face_locations) == 0:
        return None
    
    # Get face encoding
    face_encodings = face_recognition.face_encodings(
        image, 
        face_locations, 
        num_jitters=num_jitters
    )
    
    if len(face_encodings) > 0:
        return face_encodings[0]
    
    return None


if __name__ == "__main__":
    # Example usage
    print("Preprocessing training data...")
    processed_images, processed_labels = preprocess_faces(config.TRAIN_DIR)
    
    print(f"\nPreprocessed {len(processed_images)} face images")
    print(f"Image shape: {processed_images[0].shape}")
    print(f"Number of unique labels: {len(np.unique(processed_labels))}")

