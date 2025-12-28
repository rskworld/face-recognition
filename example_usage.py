"""
Example Usage Script for Face Recognition Dataset

This script demonstrates how to use the face recognition system.

Project Information:
- Project ID: 22
- Title: Face Recognition Dataset
- Category: Image Data
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
import cv2
import numpy as np
import config
from scripts.load_dataset import FaceDatasetLoader
from scripts.preprocess import preprocess_faces, get_face_encoding
from scripts.recognize_faces import FaceRecognizer


def example_load_dataset():
    """Example: Loading the dataset."""
    print("\n" + "=" * 60)
    print("Example 1: Loading Dataset")
    print("=" * 60)
    
    loader = FaceDatasetLoader(config.TRAIN_DIR)
    images, labels, label_mapping = loader.load()
    
    print(f"Loaded {len(images)} images")
    print(f"Number of identities: {len(label_mapping)}")
    print("\nLabel mapping:")
    for label, name in sorted(label_mapping.items()):
        print(f"  {label}: {name}")
    
    # Get statistics
    stats = loader.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        if key != 'images_per_person':
            print(f"  {key}: {value}")


def example_preprocess():
    """Example: Preprocessing images."""
    print("\n" + "=" * 60)
    print("Example 2: Preprocessing Images")
    print("=" * 60)
    
    processed_images, processed_labels = preprocess_faces(config.TRAIN_DIR)
    
    print(f"\nPreprocessed {len(processed_images)} face images")
    print(f"Image shape: {processed_images[0].shape}")
    print(f"Image dtype: {processed_images[0].dtype}")
    print(f"Number of unique labels: {len(np.unique(processed_labels))}")


def example_train_and_recognize():
    """Example: Training and recognizing faces."""
    print("\n" + "=" * 60)
    print("Example 3: Training and Recognition")
    print("=" * 60)
    
    # Initialize recognizer
    recognizer = FaceRecognizer()
    
    # Train the model
    print("\nTraining the model...")
    recognizer.train(config.TRAIN_DIR, save_model=True)
    
    # Example: Recognize faces in a test image (if available)
    if os.path.exists(config.TEST_DIR):
        test_images = []
        for root, dirs, files in os.walk(config.TEST_DIR):
            for file in files:
                if any(file.lower().endswith(ext) for ext in config.IMAGE_EXTENSIONS):
                    test_images.append(os.path.join(root, file))
                    break  # Just use first image found
        
        if test_images:
            test_image_path = test_images[0]
            print(f"\nRecognizing faces in: {test_image_path}")
            results = recognizer.recognize(test_image_path)
            
            for i, result in enumerate(results):
                print(f"\nFace {i + 1}:")
                print(f"  Name: {result['name']}")
                print(f"  Confidence: {result['confidence']:.2%}")
                if result['distance'] is not None:
                    print(f"  Distance: {result['distance']:.4f}")


def example_face_encoding():
    """Example: Getting face encodings."""
    print("\n" + "=" * 60)
    print("Example 4: Face Encoding")
    print("=" * 60)
    
    loader = FaceDatasetLoader(config.TRAIN_DIR)
    images, labels, label_mapping = loader.load()
    
    if len(images) > 0:
        # Get encoding for first image
        encoding = get_face_encoding(images[0])
        
        if encoding is not None:
            print(f"Face encoding shape: {encoding.shape}")
            print(f"Face encoding dtype: {encoding.dtype}")
            print(f"Encoding sample (first 5 values): {encoding[:5]}")
        else:
            print("No face detected in the first image")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Face Recognition Dataset - Example Usage")
    print("=" * 60)
    print(f"Project: Face Recognition Dataset (ID: 22)")
    print(f"RSK World - https://rskworld.in/")
    print("=" * 60)
    
    # Check if training directory exists
    if not os.path.exists(config.TRAIN_DIR):
        print(f"\nError: Training directory not found: {config.TRAIN_DIR}")
        print("Please ensure the dataset is properly organized.")
        return
    
    try:
        example_load_dataset()
        example_preprocess()
        example_face_encoding()
        example_train_and_recognize()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

