"""
Dataset Setup Script

This script helps set up the dataset directory structure.

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
import config


def create_directory_structure():
    """Create the necessary directory structure for the dataset."""
    directories = [
        config.DATA_DIR,
        config.TRAIN_DIR,
        config.TEST_DIR,
        config.VALIDATION_DIR,
        config.MODELS_DIR
    ]
    
    print("Creating directory structure...")
    print("-" * 60)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/Verified: {directory}")
    
    print("\n" + "=" * 60)
    print("Directory structure created successfully!")
    print("=" * 60)
    print("\nDataset Structure:")
    print("  data/")
    print("    train/          - Training images (person_name/image.jpg)")
    print("    test/           - Test images")
    print("    validation/     - Validation images")
    print("  models/           - Saved models")
    print("\nTo use this dataset:")
    print("  1. Organize your images in data/train/person_name/")
    print("  2. Each person should have their own folder")
    print("  3. Supported formats: JPG, JPEG, PNG")
    print("  4. Run train_model.py to train the model")


if __name__ == "__main__":
    print("=" * 60)
    print("Face Recognition Dataset Setup")
    print("=" * 60)
    print(f"Project: Face Recognition Dataset (ID: 22)")
    print(f"RSK World - https://rskworld.in/")
    print("=" * 60)
    print()
    
    create_directory_structure()

