# Face Recognition Dataset

<!--
Project Information:
- Project ID: 22
- Title: Face Recognition Dataset
- Category: Image Data
- Description: Facial recognition dataset with labeled face images across multiple identities for face recognition and verification systems.
- Full Description: This dataset contains labeled face images with multiple images per identity, various poses, lighting conditions, and expressions. Perfect for face recognition, face verification, and biometric authentication systems.
- Technologies: PNG, JPG, NumPy, OpenCV, Face Recognition
- Difficulty: Intermediate
- Source Link: ./face-recognition/face-recognition.zip
- Demo Link: ./face-recognition/

Contact Information:
RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277
Address: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in/
Year: 2026
-->

## Overview

This dataset contains labeled face images with multiple images per identity, various poses, lighting conditions, and expressions. Perfect for face recognition, face verification, and biometric authentication systems.

## Features

### Core Features
- Multiple identities
- Various poses
- Different lighting conditions
- Facial landmarks
- Ready for face recognition models

### Advanced Features
- **Face Verification**: 1:1 face matching with confidence scores
- **Face Clustering**: Automatic grouping of similar faces
- **Quality Assessment**: Image quality scoring (blur, brightness, size)
- **Face Alignment**: Automatic face alignment for better accuracy
- **Data Augmentation**: Rotate, flip, brightness, contrast adjustments
- **Batch Processing**: Process multiple images efficiently
- **REST API**: Web API for face recognition services
- **Real-time Recognition**: Webcam-based face recognition

## Technologies Used

- **PNG/JPG**: Image formats
- **NumPy**: Numerical computing
- **OpenCV**: Computer vision library
- **Face Recognition**: Face recognition library

## Dataset Structure

```
face-recognition/
├── data/
│   ├── train/
│   │   ├── person_001/
│   │   ├── person_002/
│   │   └── ...
│   ├── test/
│   │   ├── person_001/
│   │   ├── person_002/
│   │   └── ...
│   └── validation/
├── models/
├── scripts/
│   ├── load_dataset.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── recognize_faces.py
├── requirements.txt
├── config.py
└── README.md
```

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Create Sample Data

```bash
python create_sample_data.py
```

### 2. Train the Model

```bash
python train_model.py
```

### 3. Run Tests

```bash
python test_system.py
```

### 4. Try Examples

```bash
python example_usage.py
python advanced_demo.py
python demo.py
```

## Usage

### Basic Face Recognition

```python
from scripts.recognize_faces import FaceRecognizer

recognizer = FaceRecognizer()
recognizer.train('data/train', save_model=True)
results = recognizer.recognize('path/to/image.jpg')
```

### Face Verification (1:1 Matching)

```python
from scripts.advanced_features import FaceVerifier

verifier = FaceVerifier()
result = verifier.verify_images('image1.jpg', 'image2.jpg')
print(f"Match: {result['is_match']}, Confidence: {result['confidence']:.2%}")
```

### Face Clustering

```python
from scripts.advanced_features import FaceClustering
from scripts.preprocess import get_face_encoding

clusterer = FaceClustering()
encodings = [get_face_encoding(img) for img in images]
result = clusterer.cluster(encodings)
```

### Quality Assessment

```python
from scripts.advanced_features import FaceQualityAssessment

assessor = FaceQualityAssessment()
quality = assessor.assess(image, face_location)
print(f"Overall quality: {quality['overall_score']:.2f}")
```

### Data Augmentation

```python
from scripts.data_augmentation import FaceAugmenter

augmenter = FaceAugmenter()
augmented = augmenter.augment_image(image, ['rotate', 'flip', 'brightness'])
```

### REST API Server

```bash
python scripts/api_server.py
```

Then use the API endpoints:
- `POST /api/recognize` - Recognize faces
- `POST /api/verify` - Verify two faces
- `POST /api/quality` - Assess image quality
- `POST /api/cluster` - Cluster face encodings

## Requirements

See `requirements.txt` for the complete list of dependencies.

## License

This dataset is provided for educational and research purposes.

## Contact

**RSK World**
- Founder: Molla Samser
- Designer & Tester: Rima Khatun
- Email: help@rskworld.in
- Phone: +91 93305 39277
- Address: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
- Website: https://rskworld.in/

© 2026 RSK World. All rights reserved.

