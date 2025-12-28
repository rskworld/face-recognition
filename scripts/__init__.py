"""
Face Recognition Dataset Scripts Package

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

from .load_dataset import FaceDatasetLoader
from .preprocess import preprocess_faces, get_face_encoding, detect_faces
from .recognize_faces import FaceRecognizer

# Advanced features (optional imports)
try:
    from .advanced_features import (
        FaceVerifier, FaceClustering, FaceQualityAssessment,
        FaceAlignment, BatchProcessor
    )
    __all__ = [
        'FaceDatasetLoader',
        'preprocess_faces',
        'get_face_encoding',
        'detect_faces',
        'FaceRecognizer',
        'FaceVerifier',
        'FaceClustering',
        'FaceQualityAssessment',
        'FaceAlignment',
        'BatchProcessor'
    ]
except ImportError:
    __all__ = [
        'FaceDatasetLoader',
        'preprocess_faces',
        'get_face_encoding',
        'detect_faces',
        'FaceRecognizer'
    ]

