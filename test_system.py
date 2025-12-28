"""
Comprehensive Test Suite for Face Recognition System

This script tests all components of the face recognition system.

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
import sys
import numpy as np
import cv2
import config
from scripts.load_dataset import FaceDatasetLoader
from scripts.preprocess import preprocess_faces, get_face_encoding, detect_faces
from scripts.recognize_faces import FaceRecognizer
from scripts.advanced_features import (
    FaceVerifier, FaceClustering, FaceQualityAssessment, 
    FaceAlignment, BatchProcessor
)
from scripts.data_augmentation import FaceAugmenter


def test_dataset_loading():
    """Test dataset loading functionality."""
    print("\n" + "=" * 60)
    print("Test 1: Dataset Loading")
    print("=" * 60)
    
    try:
        loader = FaceDatasetLoader(config.TRAIN_DIR)
        images, labels, label_mapping = loader.load()
        
        assert len(images) > 0, "No images loaded"
        assert len(labels) > 0, "No labels loaded"
        assert len(label_mapping) > 0, "No label mapping"
        
        stats = loader.get_statistics()
        print(f"✓ Loaded {stats['total_images']} images")
        print(f"✓ Found {stats['total_identities']} identities")
        print("✓ Dataset loading: PASSED")
        return True
    except Exception as e:
        print(f"✗ Dataset loading: FAILED - {e}")
        return False


def test_face_detection():
    """Test face detection functionality."""
    print("\n" + "=" * 60)
    print("Test 2: Face Detection")
    print("=" * 60)
    
    try:
        loader = FaceDatasetLoader(config.TRAIN_DIR)
        images, labels, label_mapping = loader.load()
        
        if len(images) == 0:
            print("✗ No images available for testing")
            return False
        
        test_image = images[0]
        face_locations = detect_faces(test_image)
        
        print(f"✓ Detected {len(face_locations)} face(s) in test image")
        print("✓ Face detection: PASSED")
        return True
    except Exception as e:
        print(f"✗ Face detection: FAILED - {e}")
        return False


def test_face_encoding():
    """Test face encoding functionality."""
    print("\n" + "=" * 60)
    print("Test 3: Face Encoding")
    print("=" * 60)
    
    try:
        loader = FaceDatasetLoader(config.TRAIN_DIR)
        images, labels, label_mapping = loader.load()
        
        if len(images) == 0:
            print("✗ No images available for testing")
            return False
        
        test_image = images[0]
        encoding = get_face_encoding(test_image)
        
        assert encoding is not None, "No encoding generated"
        assert encoding.shape == (128,), f"Wrong encoding shape: {encoding.shape}"
        
        print(f"✓ Generated encoding with shape: {encoding.shape}")
        print("✓ Face encoding: PASSED")
        return True
    except Exception as e:
        print(f"✗ Face encoding: FAILED - {e}")
        return False


def test_face_recognition():
    """Test face recognition functionality."""
    print("\n" + "=" * 60)
    print("Test 4: Face Recognition")
    print("=" * 60)
    
    try:
        recognizer = FaceRecognizer()
        
        if not os.path.exists(recognizer.model_path):
            print("Training model for testing...")
            recognizer.train(config.TRAIN_DIR, save_model=True)
        else:
            recognizer.load_model()
        
        assert len(recognizer.known_face_encodings) > 0, "No encodings in model"
        assert len(recognizer.known_face_names) > 0, "No names in model"
        
        print(f"✓ Model loaded with {len(recognizer.known_face_encodings)} encodings")
        print("✓ Face recognition: PASSED")
        return True
    except Exception as e:
        print(f"✗ Face recognition: FAILED - {e}")
        return False


def test_face_verification():
    """Test face verification functionality."""
    print("\n" + "=" * 60)
    print("Test 5: Face Verification")
    print("=" * 60)
    
    try:
        verifier = FaceVerifier()
        
        # Create two test encodings
        encoding1 = np.random.rand(128)
        encoding2 = encoding1 + np.random.rand(128) * 0.1  # Similar
        encoding3 = np.random.rand(128)  # Different
        
        result1 = verifier.verify(encoding1, encoding2)
        result2 = verifier.verify(encoding1, encoding3)
        
        assert 'is_match' in result1, "Missing is_match in result"
        assert 'distance' in result1, "Missing distance in result"
        assert 'confidence' in result1, "Missing confidence in result"
        
        print(f"✓ Verification test 1 (similar): match={result1['is_match']}, confidence={result1['confidence']:.2f}")
        print(f"✓ Verification test 2 (different): match={result2['is_match']}, confidence={result2['confidence']:.2f}")
        print("✓ Face verification: PASSED")
        return True
    except Exception as e:
        print(f"✗ Face verification: FAILED - {e}")
        return False


def test_face_clustering():
    """Test face clustering functionality."""
    print("\n" + "=" * 60)
    print("Test 6: Face Clustering")
    print("=" * 60)
    
    try:
        clusterer = FaceClustering()
        
        # Create test encodings (3 groups)
        encodings = []
        for group in range(3):
            base = np.random.rand(128)
            for _ in range(3):
                encodings.append(base + np.random.rand(128) * 0.2)
        
        result = clusterer.cluster(encodings)
        
        assert 'labels' in result, "Missing labels in result"
        assert 'n_clusters' in result, "Missing n_clusters in result"
        
        print(f"✓ Clustered {len(encodings)} encodings into {result['n_clusters']} clusters")
        print("✓ Face clustering: PASSED")
        return True
    except Exception as e:
        print(f"✗ Face clustering: FAILED - {e}")
        return False


def test_quality_assessment():
    """Test quality assessment functionality."""
    print("\n" + "=" * 60)
    print("Test 7: Quality Assessment")
    print("=" * 60)
    
    try:
        assessor = FaceQualityAssessment()
        
        # Create test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        face_location = (50, 150, 150, 50)
        
        quality = assessor.assess(test_image, face_location)
        
        assert 'blur_score' in quality, "Missing blur_score"
        assert 'brightness_score' in quality, "Missing brightness_score"
        assert 'overall_score' in quality, "Missing overall_score"
        
        print(f"✓ Quality scores: blur={quality['blur_score']:.2f}, brightness={quality['brightness_score']:.2f}")
        print(f"✓ Overall score: {quality['overall_score']:.2f}")
        print("✓ Quality assessment: PASSED")
        return True
    except Exception as e:
        print(f"✗ Quality assessment: FAILED - {e}")
        return False


def test_data_augmentation():
    """Test data augmentation functionality."""
    print("\n" + "=" * 60)
    print("Test 8: Data Augmentation")
    print("=" * 60)
    
    try:
        augmenter = FaceAugmenter()
        
        # Create test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Test individual augmentations
        rotated = augmenter.rotate(test_image, 15)
        flipped = augmenter.flip_horizontal(test_image)
        bright = augmenter.adjust_brightness(test_image, 0.2)
        
        assert rotated.shape == test_image.shape, "Rotation changed shape"
        assert flipped.shape == test_image.shape, "Flip changed shape"
        assert bright.shape == test_image.shape, "Brightness changed shape"
        
        print("✓ Rotation augmentation: PASSED")
        print("✓ Flip augmentation: PASSED")
        print("✓ Brightness augmentation: PASSED")
        print("✓ Data augmentation: PASSED")
        return True
    except Exception as e:
        print(f"✗ Data augmentation: FAILED - {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Face Recognition System - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Project: Face Recognition Dataset (ID: 22)")
    print(f"RSK World - https://rskworld.in/")
    print("=" * 60)
    
    # Check if sample data exists
    if not os.path.exists(config.TRAIN_DIR) or len(os.listdir(config.TRAIN_DIR)) == 0:
        print("\n⚠ Sample data not found. Creating sample data...")
        from create_sample_data import create_synthetic_faces
        create_synthetic_faces()
    
    tests = [
        test_dataset_loading,
        test_face_detection,
        test_face_encoding,
        test_face_recognition,
        test_face_verification,
        test_face_clustering,
        test_quality_assessment,
        test_data_augmentation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print("=" * 60)
    
    if passed == total:
        print("✓ All tests PASSED!")
        return 0
    else:
        print("✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

