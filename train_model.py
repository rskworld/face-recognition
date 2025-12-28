"""
Main Training Script for Face Recognition Model

This script trains a face recognition model on the dataset.

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
import sys
import config
from scripts.recognize_faces import FaceRecognizer
from scripts.load_dataset import FaceDatasetLoader


def main():
    """Main training function."""
    print("=" * 60)
    print("Face Recognition Model Training")
    print("=" * 60)
    print(f"Project: Face Recognition Dataset (ID: 22)")
    print(f"RSK World - https://rskworld.in/")
    print("=" * 60)
    print()
    
    # Check if training directory exists
    if not os.path.exists(config.TRAIN_DIR):
        print(f"Error: Training directory not found: {config.TRAIN_DIR}")
        print("Please ensure the dataset is properly organized.")
        print("\nExpected structure:")
        print("  data/train/person_name/image.jpg")
        sys.exit(1)
    
    # Load dataset statistics
    print("Loading dataset...")
    loader = FaceDatasetLoader(config.TRAIN_DIR)
    stats = loader.get_statistics()
    
    print("\nDataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total identities: {stats['total_identities']}")
    print(f"  Average images per person: {stats['average_images_per_person']:.2f}")
    print()
    
    # Initialize recognizer
    recognizer = FaceRecognizer(tolerance=config.TOLERANCE)
    
    # Train the model
    print("Starting training...")
    recognizer.train(config.TRAIN_DIR, save_model=config.SAVE_MODEL)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {recognizer.model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

