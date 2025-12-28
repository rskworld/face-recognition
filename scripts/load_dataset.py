"""
Dataset Loader for Face Recognition Dataset

This script loads and organizes face images from the dataset directory structure.

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
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Dict
import config


class FaceDatasetLoader:
    """
    Loads face images from organized directory structure.
    Expected structure: data/train/person_name/image.jpg
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Path to the dataset directory (default: config.TRAIN_DIR)
        """
        self.data_dir = data_dir or config.TRAIN_DIR
        self.images = []
        self.labels = []
        self.label_to_name = {}
        self.name_to_label = {}
        
    def load(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load all images and labels from the dataset.
        
        Returns:
            Tuple of (images, labels, label_mapping)
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        
        label_counter = 0
        
        # Iterate through each person's directory
        for person_name in sorted(os.listdir(self.data_dir)):
            person_dir = os.path.join(self.data_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
            
            # Assign label to person
            if person_name not in self.name_to_label:
                self.name_to_label[person_name] = label_counter
                self.label_to_name[label_counter] = person_name
                label_counter += 1
            
            label = self.name_to_label[person_name]
            
            # Load all images for this person
            for filename in os.listdir(person_dir):
                if any(filename.lower().endswith(ext) for ext in config.IMAGE_EXTENSIONS):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        image = cv2.imread(image_path)
                        if image is not None:
                            # Convert BGR to RGB
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            self.images.append(image)
                            self.labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
        
        return np.array(self.images), np.array(self.labels), self.label_to_name
    
    def get_person_images(self, person_name: str) -> List[np.ndarray]:
        """
        Get all images for a specific person.
        
        Args:
            person_name: Name of the person
            
        Returns:
            List of images for that person
        """
        person_dir = os.path.join(self.data_dir, person_name)
        images = []
        
        if os.path.exists(person_dir):
            for filename in os.listdir(person_dir):
                if any(filename.lower().endswith(ext) for ext in config.IMAGE_EXTENSIONS):
                    image_path = os.path.join(person_dir, filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images.append(image)
        
        return images
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if len(self.images) == 0:
            self.load()
        
        unique_labels = len(set(self.labels))
        total_images = len(self.images)
        
        # Count images per person
        images_per_person = {}
        for label, name in self.label_to_name.items():
            count = np.sum(self.labels == label)
            images_per_person[name] = count
        
        return {
            'total_images': total_images,
            'total_identities': unique_labels,
            'images_per_person': images_per_person,
            'average_images_per_person': total_images / unique_labels if unique_labels > 0 else 0
        }


if __name__ == "__main__":
    # Example usage
    loader = FaceDatasetLoader(config.TRAIN_DIR)
    images, labels, label_mapping = loader.load()
    
    print(f"Loaded {len(images)} images")
    print(f"Number of identities: {len(label_mapping)}")
    print("\nLabel mapping:")
    for label, name in label_mapping.items():
        print(f"  {label}: {name}")
    
    stats = loader.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total identities: {stats['total_identities']}")
    print(f"  Average images per person: {stats['average_images_per_person']:.2f}")

