# Quick Start Guide

<!--
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
-->

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up directory structure:**
   ```bash
   python setup_dataset.py
   ```

## Dataset Organization

Organize your images in the following structure:

```
data/
  train/
    person_001/
      image1.jpg
      image2.jpg
      ...
    person_002/
      image1.jpg
      image2.jpg
      ...
```

## Basic Usage

### 1. Train the Model

```bash
python train_model.py
```

### 2. Run Examples

```bash
python example_usage.py
```

### 3. Interactive Demo

```bash
python demo.py
```

## Python API

### Load Dataset

```python
from scripts.load_dataset import FaceDatasetLoader

loader = FaceDatasetLoader('data/train')
images, labels, label_mapping = loader.load()
```

### Preprocess Images

```python
from scripts.preprocess import preprocess_faces

processed_images, processed_labels = preprocess_faces('data/train')
```

### Train and Recognize

```python
from scripts.recognize_faces import FaceRecognizer

# Train
recognizer = FaceRecognizer()
recognizer.train('data/train', save_model=True)

# Recognize
results = recognizer.recognize('path/to/image.jpg')
for result in results:
    print(f"Name: {result['name']}, Confidence: {result['confidence']:.2%}")
```

## Features

- Multiple identities support
- Various poses handling
- Different lighting conditions
- Facial landmarks detection
- Ready for face recognition models

## Support

For questions or support, contact:
- Email: help@rskworld.in
- Website: https://rskworld.in/

© 2026 RSK World. All rights reserved.

