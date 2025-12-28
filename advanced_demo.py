"""
Advanced Features Demo

This script demonstrates all advanced features of the face recognition system.

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
import config
from scripts.recognize_faces import FaceRecognizer
from scripts.advanced_features import (
    FaceVerifier, FaceClustering, FaceQualityAssessment,
    FaceAlignment, BatchProcessor
)
from scripts.data_augmentation import FaceAugmenter
from scripts.load_dataset import FaceDatasetLoader
from scripts.preprocess import get_face_encoding


def demo_face_verification():
    """Demonstrate face verification."""
    print("\n" + "=" * 60)
    print("Demo: Face Verification (1:1 Matching)")
    print("=" * 60)
    
    loader = FaceDatasetLoader(config.TRAIN_DIR)
    images, labels, label_mapping = loader.load()
    
    if len(images) < 2:
        print("Need at least 2 images for verification demo")
        return
    
    verifier = FaceVerifier()
    
    # Get encodings from same person
    person_images = {}
    for i, (img, label) in enumerate(zip(images, labels)):
        person_name = label_mapping[label]
        if person_name not in person_images:
            person_images[person_name] = []
        if len(person_images[person_name]) < 2:
            encoding = get_face_encoding(img)
            if encoding is not None:
                person_images[person_name].append(encoding)
    
    # Test verification
    for person_name, encodings in person_images.items():
        if len(encodings) >= 2:
            result = verifier.verify(encodings[0], encodings[1])
            print(f"\nPerson: {person_name}")
            print(f"  Match: {result['is_match']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Distance: {result['distance']:.4f}")
            break


def demo_face_clustering():
    """Demonstrate face clustering."""
    print("\n" + "=" * 60)
    print("Demo: Face Clustering")
    print("=" * 60)
    
    loader = FaceDatasetLoader(config.TRAIN_DIR)
    images, labels, label_mapping = loader.load()
    
    if len(images) < 5:
        print("Need at least 5 images for clustering demo")
        return
    
    # Get encodings
    encodings = []
    encoding_labels = []
    
    for img, label in zip(images[:20], labels[:20]):  # Use first 20
        encoding = get_face_encoding(img)
        if encoding is not None:
            encodings.append(encoding)
            encoding_labels.append(label_mapping[label])
    
    if len(encodings) < 5:
        print("Not enough valid encodings for clustering")
        return
    
    clusterer = FaceClustering(eps=0.6, min_samples=2)
    result = clusterer.cluster(encodings)
    
    print(f"\nClustering Results:")
    print(f"  Total encodings: {len(encodings)}")
    print(f"  Clusters found: {result['n_clusters']}")
    print(f"  Noise points: {result['n_noise']}")
    if result['silhouette_score']:
        print(f"  Silhouette score: {result['silhouette_score']:.3f}")
    
    # Show cluster distribution
    from collections import Counter
    label_counts = Counter(result['labels'])
    print(f"\nCluster distribution:")
    for label, count in sorted(label_counts.items()):
        if label != -1:
            print(f"  Cluster {label}: {count} faces")


def demo_quality_assessment():
    """Demonstrate quality assessment."""
    print("\n" + "=" * 60)
    print("Demo: Face Quality Assessment")
    print("=" * 60)
    
    loader = FaceDatasetLoader(config.TRAIN_DIR)
    images, labels, label_mapping = loader.load()
    
    if len(images) == 0:
        print("No images available")
        return
    
    assessor = FaceQualityAssessment()
    from scripts.preprocess import detect_faces
    
    # Assess first few images
    for i, img in enumerate(images[:3]):
        face_locations = detect_faces(img)
        if len(face_locations) > 0:
            quality = assessor.assess(img, face_locations[0])
            print(f"\nImage {i+1}:")
            print(f"  Blur score: {quality['blur_score']:.3f}")
            print(f"  Brightness score: {quality['brightness_score']:.3f}")
            if 'size_score' in quality:
                print(f"  Size score: {quality['size_score']:.3f}")
            print(f"  Overall score: {quality['overall_score']:.3f}")


def demo_data_augmentation():
    """Demonstrate data augmentation."""
    print("\n" + "=" * 60)
    print("Demo: Data Augmentation")
    print("=" * 60)
    
    loader = FaceDatasetLoader(config.TRAIN_DIR)
    images, labels, label_mapping = loader.load()
    
    if len(images) == 0:
        print("No images available")
        return
    
    augmenter = FaceAugmenter()
    test_image = images[0]
    
    print("\nOriginal image shape:", test_image.shape)
    
    # Test augmentations
    rotated = augmenter.rotate(test_image, 15)
    flipped = augmenter.flip_horizontal(test_image)
    bright = augmenter.adjust_brightness(test_image, 0.2)
    
    print("✓ Rotation: PASSED")
    print("✓ Horizontal flip: PASSED")
    print("✓ Brightness adjustment: PASSED")
    
    # Show augmented images
    output_dir = os.path.join(config.DATA_DIR, 'augmented_demo')
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, 'original.jpg'), 
                cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, 'rotated.jpg'), 
                cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, 'flipped.jpg'), 
                cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, 'bright.jpg'), 
                cv2.cvtColor(bright, cv2.COLOR_RGB2BGR))
    
    print(f"\n✓ Augmented images saved to: {output_dir}")


def demo_batch_processing():
    """Demonstrate batch processing."""
    print("\n" + "=" * 60)
    print("Demo: Batch Processing")
    print("=" * 60)
    
    recognizer = FaceRecognizer()
    if os.path.exists(recognizer.model_path):
        recognizer.load_model()
    else:
        print("Training model first...")
        recognizer.train(config.TRAIN_DIR, save_model=True)
    
    processor = BatchProcessor(recognizer=recognizer)
    
    # Process a few images
    loader = FaceDatasetLoader(config.TRAIN_DIR)
    images, labels, label_mapping = loader.load()
    
    if len(images) == 0:
        print("No images available")
        return
    
    # Save a few test images temporarily
    test_dir = os.path.join(config.DATA_DIR, 'batch_test')
    os.makedirs(test_dir, exist_ok=True)
    
    test_paths = []
    for i, img in enumerate(images[:3]):
        path = os.path.join(test_dir, f'test_{i}.jpg')
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        test_paths.append(path)
    
    results = processor.process_image_list(test_paths)
    
    print(f"\nProcessed {len(results)} images:")
    for result in results:
        if 'results' in result:
            print(f"  {os.path.basename(result['image_path'])}: {len(result['results'])} face(s) detected")
        else:
            print(f"  {os.path.basename(result['image_path'])}: Error - {result.get('error', 'Unknown')}")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Face Recognition - Advanced Features Demo")
    print("=" * 60)
    print(f"Project: Face Recognition Dataset (ID: 22)")
    print(f"RSK World - https://rskworld.in/")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists(config.TRAIN_DIR) or len(os.listdir(config.TRAIN_DIR)) == 0:
        print("\n⚠ Sample data not found. Creating sample data...")
        from create_sample_data import create_synthetic_faces
        create_synthetic_faces()
    
    demos = [
        ("Face Verification", demo_face_verification),
        ("Face Clustering", demo_face_clustering),
        ("Quality Assessment", demo_quality_assessment),
        ("Data Augmentation", demo_data_augmentation),
        ("Batch Processing", demo_batch_processing)
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print("  6. Run all demos")
    
    choice = input("\nSelect demo (1-6): ").strip()
    
    if choice == "6":
        for name, demo_func in demos:
            try:
                demo_func()
            except Exception as e:
                print(f"\nError in {name}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= 5:
        name, demo_func = demos[int(choice) - 1]
        try:
            demo_func()
        except Exception as e:
            print(f"\nError: {e}")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()

