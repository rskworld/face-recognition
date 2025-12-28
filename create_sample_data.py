"""
Create Sample Data for Face Recognition Dataset

This script creates sample data using publicly available resources for educational purposes.

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

import os
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import config


def create_synthetic_faces():
    """
    Create synthetic face images for testing.
    Uses OpenCV to generate simple face-like patterns for educational purposes.
    """
    os.makedirs(config.TRAIN_DIR, exist_ok=True)
    
    # Create sample identities
    identities = ['person_001', 'person_002', 'person_003', 'person_004', 'person_005']
    
    print("Creating sample face images...")
    print("-" * 60)
    
    for person_id in identities:
        person_dir = os.path.join(config.TRAIN_DIR, person_id)
        os.makedirs(person_dir, exist_ok=True)
        
        # Create 5 variations per person
        for i in range(5):
            # Create a simple face-like pattern
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            
            # Face shape (oval)
            cv2.ellipse(img, (100, 100), (80, 100), 0, 0, 360, (220, 180, 140), -1)
            
            # Eyes
            cv2.circle(img, (75, 80), 10, (0, 0, 0), -1)
            cv2.circle(img, (125, 80), 10, (0, 0, 0), -1)
            
            # Nose
            cv2.ellipse(img, (100, 110), (5, 15), 0, 0, 360, (180, 140, 100), -1)
            
            # Mouth
            cv2.ellipse(img, (100, 140), (20, 10), 0, 0, 180, (100, 50, 50), 2)
            
            # Add some variation
            noise = np.random.randint(-20, 20, (200, 200, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Add slight rotation variation
            if i > 0:
                angle = np.random.uniform(-5, 5)
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # Save image
            filename = f"{person_id}_{i+1:02d}.jpg"
            filepath = os.path.join(person_dir, filename)
            cv2.imwrite(filepath, img)
            print(f"Created: {filepath}")
    
    print("\n" + "=" * 60)
    print("Sample data created successfully!")
    print(f"Created {len(identities)} identities with 5 images each")
    print("=" * 60)


def download_sample_images():
    """
    Download sample images from publicly available sources.
    Uses placeholder images for educational purposes.
    """
    # Note: In a real scenario, you would download from a public dataset
    # For educational purposes, we'll create synthetic data instead
    print("Using synthetic data generation for educational purposes...")
    create_synthetic_faces()


if __name__ == "__main__":
    print("=" * 60)
    print("Face Recognition Dataset - Sample Data Creator")
    print("=" * 60)
    print(f"Project: Face Recognition Dataset (ID: 22)")
    print(f"RSK World - https://rskworld.in/")
    print("=" * 60)
    print()
    
    # Create sample data
    create_synthetic_faces()
    
    print("\nNext steps:")
    print("1. Run: python train_model.py")
    print("2. Run: python example_usage.py")
    print("3. Run: python demo.py")

