# Installation and Setup Guide

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

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Windows/Linux/macOS

## Step-by-Step Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** On some systems, you may need to install `dlib` separately:

**Windows:**
```bash
pip install dlib
```

**Linux/macOS:**
```bash
# Install cmake first
sudo apt-get install cmake  # Linux
brew install cmake  # macOS

pip install dlib
```

### 2. Set Up Directory Structure

```bash
python setup_dataset.py
```

This creates the necessary directories:
- `data/train/` - Training images
- `data/test/` - Test images
- `data/validation/` - Validation images
- `models/` - Saved models

### 3. Create Sample Data (Optional)

For testing purposes, create sample data:

```bash
python create_sample_data.py
```

This creates 5 sample identities with 5 images each (25 total images).

### 4. Train the Model

```bash
python train_model.py
```

This will:
- Load all images from `data/train/`
- Extract face encodings
- Save the trained model to `models/face_recognition_model.pkl`

### 5. Test the System

```bash
python test_system.py
```

This runs comprehensive tests on all components.

### 6. Run Examples

```bash
# Basic examples
python example_usage.py

# Advanced features demo
python advanced_demo.py

# Interactive demo (webcam/image)
python demo.py
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'face_recognition'`

**Solution:**
```bash
pip install face-recognition
```

### Issue: `dlib` installation fails

**Solution:**
- Windows: Download pre-built wheel from https://github.com/sachadee/Dlib
- Linux: Install cmake and build tools first
- macOS: Install Xcode command line tools

### Issue: OpenCV not found

**Solution:**
```bash
pip install opencv-python
```

### Issue: No faces detected

**Possible causes:**
1. Images don't contain clear faces
2. Face detection model needs adjustment
3. Image quality is too low

**Solution:**
- Use clear, front-facing images
- Adjust `FACE_DETECTION_MODEL` in `config.py` (try 'cnn' instead of 'hog')
- Ensure images are in RGB format

## Verification

After installation, verify everything works:

```python
import face_recognition
import cv2
import numpy as np
from scripts.load_dataset import FaceDatasetLoader

# Should work without errors
print("✓ All imports successful")
```

## Next Steps

1. **Add Your Own Data:**
   - Organize images in `data/train/person_name/image.jpg`
   - Each person should have their own folder
   - Use clear, front-facing images for best results

2. **Train Custom Model:**
   ```bash
   python train_model.py
   ```

3. **Use the API:**
   ```bash
   python scripts/api_server.py
   ```
   Then access at `http://localhost:5000`

## Support

For issues or questions:
- Email: help@rskworld.in
- Website: https://rskworld.in/

© 2026 RSK World. All rights reserved.

