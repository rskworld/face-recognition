"""
Advanced Face Recognition Features

This module provides advanced features like face verification, clustering, 
quality assessment, and batch processing.

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
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import config
from scripts.preprocess import get_face_encoding, detect_faces


class FaceVerifier:
    """
    Face verification system for 1:1 face matching.
    """
    
    def __init__(self, threshold: float = None):
        """
        Initialize face verifier.
        
        Args:
            threshold: Distance threshold for verification (default from config)
        """
        self.threshold = threshold or config.TOLERANCE
    
    def verify(self, encoding1: np.ndarray, encoding2: np.ndarray) -> Dict:
        """
        Verify if two face encodings belong to the same person.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            Dictionary with verification result and confidence
        """
        distance = np.linalg.norm(encoding1 - encoding2)
        is_match = distance <= self.threshold
        confidence = max(0, 1 - (distance / self.threshold)) if is_match else 0
        
        return {
            'is_match': is_match,
            'distance': float(distance),
            'confidence': float(confidence),
            'threshold': self.threshold
        }
    
    def verify_images(self, image1_path: str, image2_path: str) -> Dict:
        """
        Verify if two images contain the same person.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Verification result
        """
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        if img1 is None or img2 is None:
            return {'error': 'Could not load one or both images'}
        
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        encoding1 = get_face_encoding(rgb1)
        encoding2 = get_face_encoding(rgb2)
        
        if encoding1 is None or encoding2 is None:
            return {'error': 'Could not detect face in one or both images'}
        
        return self.verify(encoding1, encoding2)


class FaceClustering:
    """
    Face clustering system to group similar faces.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        """
        Initialize face clustering.
        
        Args:
            eps: Maximum distance between samples in the same cluster
            min_samples: Minimum number of samples in a cluster
        """
        self.eps = eps
        self.min_samples = min_samples
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    
    def cluster(self, encodings: List[np.ndarray]) -> Dict:
        """
        Cluster face encodings into groups.
        
        Args:
            encodings: List of face encodings
            
        Returns:
            Dictionary with cluster labels and statistics
        """
        if len(encodings) == 0:
            return {'error': 'No encodings provided'}
        
        encodings_array = np.array(encodings)
        cluster_labels = self.clusterer.fit_predict(encodings_array)
        
        # Calculate statistics
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Group by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        # Calculate silhouette score if possible
        silhouette = None
        if n_clusters > 1 and len(encodings) > 1:
            try:
                silhouette = float(silhouette_score(encodings_array, cluster_labels))
            except:
                pass
        
        return {
            'labels': cluster_labels.tolist(),
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'clusters': clusters,
            'silhouette_score': silhouette
        }


class FaceQualityAssessment:
    """
    Assess the quality of face images for recognition.
    """
    
    @staticmethod
    def assess_blur(image: np.ndarray) -> float:
        """
        Assess image blur using Laplacian variance.
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            Blur score (higher = less blur)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)
    
    @staticmethod
    def assess_brightness(image: np.ndarray) -> float:
        """
        Assess image brightness.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Brightness score (0-1, 0.5 is ideal)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        mean_brightness = np.mean(gray) / 255.0
        # Score is better when closer to 0.5
        brightness_score = 1 - abs(mean_brightness - 0.5) * 2
        return float(brightness_score)
    
    @staticmethod
    def assess_face_size(image: np.ndarray, face_location: Tuple) -> float:
        """
        Assess if face size is appropriate.
        
        Args:
            image: Input image
            face_location: Face location (top, right, bottom, left)
            
        Returns:
            Size score (0-1)
        """
        top, right, bottom, left = face_location
        face_height = bottom - top
        face_width = right - left
        image_height, image_width = image.shape[:2]
        
        face_area = face_height * face_width
        image_area = image_height * image_width
        
        face_ratio = face_area / image_area
        
        # Ideal ratio is around 0.1-0.3
        if 0.1 <= face_ratio <= 0.3:
            return 1.0
        elif face_ratio < 0.1:
            return face_ratio / 0.1
        else:
            return max(0, 1 - (face_ratio - 0.3) / 0.7)
    
    def assess(self, image: np.ndarray, face_location: Tuple = None) -> Dict:
        """
        Comprehensive quality assessment.
        
        Args:
            image: Input image (RGB)
            face_location: Optional face location
            
        Returns:
            Dictionary with quality scores
        """
        blur_score = self.assess_blur(image)
        brightness_score = self.assess_brightness(image)
        
        # Normalize blur score (typical good values are > 100)
        blur_normalized = min(1.0, blur_score / 500.0)
        
        scores = {
            'blur_score': blur_normalized,
            'brightness_score': brightness_score,
            'overall_score': (blur_normalized + brightness_score) / 2
        }
        
        if face_location:
            size_score = self.assess_face_size(image, face_location)
            scores['size_score'] = size_score
            scores['overall_score'] = (blur_normalized + brightness_score + size_score) / 3
        
        return scores


class FaceAlignment:
    """
    Face alignment utilities for better recognition accuracy.
    """
    
    @staticmethod
    def align_face(image: np.ndarray, face_landmarks: Dict) -> Optional[np.ndarray]:
        """
        Align face using facial landmarks.
        
        Args:
            image: Input image (RGB)
            face_landmarks: Face landmarks dictionary from face_recognition
            
        Returns:
            Aligned face image or None
        """
        try:
            # Get eye landmarks
            left_eye = face_landmarks['left_eye']
            right_eye = face_landmarks['right_eye']
            
            # Calculate eye centers
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            
            # Calculate angle
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate center point
            eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                         (left_eye_center[1] + right_eye_center[1]) // 2)
            
            # Rotate image
            rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            aligned = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return aligned
        except Exception as e:
            print(f"Alignment error: {e}")
            return None
    
    @staticmethod
    def get_aligned_face(image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get aligned face from image.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Aligned face image or None
        """
        face_landmarks_list = face_recognition.face_landmarks(image)
        
        if len(face_landmarks_list) == 0:
            return None
        
        aligned = FaceAlignment.align_face(image, face_landmarks_list[0])
        return aligned


class BatchProcessor:
    """
    Batch processing utilities for multiple images.
    """
    
    def __init__(self, recognizer=None):
        """
        Initialize batch processor.
        
        Args:
            recognizer: FaceRecognizer instance (optional)
        """
        self.recognizer = recognizer
    
    def process_directory(self, directory: str, output_file: str = None) -> List[Dict]:
        """
        Process all images in a directory.
        
        Args:
            directory: Directory containing images
            output_file: Optional JSON output file
            
        Returns:
            List of recognition results
        """
        import os
        import json
        
        results = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in config.IMAGE_EXTENSIONS):
                    image_path = os.path.join(root, file)
                    
                    try:
                        if self.recognizer:
                            result = self.recognizer.recognize(image_path)
                            results.append({
                                'image_path': image_path,
                                'results': result
                            })
                    except Exception as e:
                        results.append({
                            'image_path': image_path,
                            'error': str(e)
                        })
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def process_image_list(self, image_paths: List[str]) -> List[Dict]:
        """
        Process a list of image paths.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of recognition results
        """
        results = []
        
        for image_path in image_paths:
            try:
                if self.recognizer:
                    result = self.recognizer.recognize(image_path)
                    results.append({
                        'image_path': image_path,
                        'results': result
                    })
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results

