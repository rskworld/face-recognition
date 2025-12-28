"""
Dataset Visualization Script

This script provides visualization utilities for the face recognition dataset.

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
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple
import config
from scripts.load_dataset import FaceDatasetLoader
from scripts.preprocess import detect_faces, extract_face


def visualize_dataset_samples(data_dir: str = None, num_samples: int = 16, 
                              save_path: str = None):
    """
    Visualize random samples from the dataset.
    
    Args:
        data_dir: Directory containing the dataset
        num_samples: Number of samples to display
        save_path: Optional path to save the visualization
    """
    if data_dir is None:
        data_dir = config.TRAIN_DIR
    
    loader = FaceDatasetLoader(data_dir)
    images, labels, label_mapping = loader.load()
    
    if len(images) == 0:
        print("No images found in the dataset")
        return
    
    # Select random samples
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # Create grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            image_idx = indices[idx]
            image = images[image_idx]
            label = labels[image_idx]
            person_name = label_mapping[label]
            
            ax.imshow(image)
            ax.set_title(f"{person_name}\n(Label: {label})", fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle('Face Recognition Dataset Samples', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def visualize_face_detections(image_path: str, save_path: str = None):
    """
    Visualize face detections on an image.
    
    Args:
        image_path: Path to the image
        save_path: Optional path to save the visualization
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = detect_faces(rgb_image)
    
    # Draw bounding boxes
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(rgb_image)
    
    for top, right, bottom, left in face_locations:
        rect = plt.Rectangle(
            (left, top), 
            right - left, 
            bottom - top,
            fill=False, 
            edgecolor='red', 
            linewidth=2
        )
        ax.add_patch(rect)
    
    ax.set_title(f'Face Detections ({len(face_locations)} faces found)', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def visualize_person_samples(person_name: str, data_dir: str = None, 
                            num_samples: int = 6, save_path: str = None):
    """
    Visualize samples from a specific person.
    
    Args:
        person_name: Name of the person
        data_dir: Directory containing the dataset
        num_samples: Number of samples to display
        save_path: Optional path to save the visualization
    """
    if data_dir is None:
        data_dir = config.TRAIN_DIR
    
    loader = FaceDatasetLoader(data_dir)
    images = loader.get_person_images(person_name)
    
    if len(images) == 0:
        print(f"No images found for person: {person_name}")
        return
    
    num_samples = min(num_samples, len(images))
    selected_images = images[:num_samples]
    
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            ax.imshow(selected_images[idx])
            ax.set_title(f"Sample {idx + 1}", fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(f'Samples from: {person_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def plot_dataset_statistics(data_dir: str = None, save_path: str = None):
    """
    Plot dataset statistics.
    
    Args:
        data_dir: Directory containing the dataset
        save_path: Optional path to save the plot
    """
    if data_dir is None:
        data_dir = config.TRAIN_DIR
    
    loader = FaceDatasetLoader(data_dir)
    stats = loader.get_statistics()
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Images per person
    images_per_person = stats['images_per_person']
    names = list(images_per_person.keys())
    counts = list(images_per_person.values())
    
    axes[0].bar(range(len(names)), counts, color='skyblue', edgecolor='navy')
    axes[0].set_xlabel('Person Index')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title('Images per Person')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([f"P{i+1}" for i in range(len(names))], rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Overall statistics
    stats_text = f"""
    Total Images: {stats['total_images']}
    Total Identities: {stats['total_identities']}
    Avg Images/Person: {stats['average_images_per_person']:.1f}
    """
    axes[1].text(0.1, 0.5, stats_text, fontsize=14, 
                 verticalalignment='center', family='monospace')
    axes[1].set_title('Dataset Statistics')
    axes[1].axis('off')
    
    plt.suptitle('Face Recognition Dataset Statistics', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Statistics plot saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Face Recognition Dataset Visualization")
    print("=" * 60)
    print(f"Project: Face Recognition Dataset (ID: 22)")
    print(f"RSK World - https://rskworld.in/")
    print("=" * 60)
    print()
    
    if not os.path.exists(config.TRAIN_DIR):
        print(f"Error: Training directory not found: {config.TRAIN_DIR}")
        sys.exit(1)
    
    print("Available visualizations:")
    print("1. Dataset samples")
    print("2. Dataset statistics")
    print("3. Person samples")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        visualize_dataset_samples()
    elif choice == "2":
        plot_dataset_statistics()
    elif choice == "3":
        person_name = input("Enter person name: ").strip()
        visualize_person_samples(person_name)
    else:
        print("Invalid choice")

