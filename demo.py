"""
Interactive Demo Script for Face Recognition

This script provides an interactive demo for face recognition.

Project Information:
- Project ID: 22
- Title: Face Recognition Dataset
- Category: Image Data
- Technologies: PNG, JPG, NumPy, OpenCV, Face Recognition
- Difficulty: Intermediate

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
import face_recognition


def draw_face_boxes(image, results):
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image: Input image (BGR format)
        results: List of recognition results
        
    Returns:
        Image with drawn boxes and labels
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for result in results:
        top, right, bottom, left = result['location']
        name = result['name']
        confidence = result['confidence']
        
        # Draw rectangle
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        
        # Draw label
        label = f"{name} ({confidence:.2%})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = top - 10 if top - 10 > 10 else top + 20
        
        cv2.rectangle(
            image,
            (left, label_y - label_size[1] - 5),
            (left + label_size[0], label_y + 5),
            color,
            -1
        )
        cv2.putText(
            image,
            label,
            (left, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return image


def recognize_from_webcam():
    """Recognize faces from webcam feed."""
    recognizer = FaceRecognizer()
    
    # Load or train model
    if os.path.exists(recognizer.model_path):
        print("Loading trained model...")
        recognizer.load_model()
    else:
        print("No trained model found. Training new model...")
        recognizer.train(config.TRAIN_DIR, save_model=True)
    
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nFace Recognition Demo - Webcam Mode")
    print("Press 'q' to quit")
    print("-" * 60)
    
    process_this_frame = True
    
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        # Process every other frame for performance
        if process_this_frame:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Recognize faces
            results = recognizer.recognize_from_array(rgb_small_frame)
            
            # Scale back up face locations
            for result in results:
                top, right, bottom, left = result['location']
                result['location'] = (
                    top * 4, right * 4, bottom * 4, left * 4
                )
        
        process_this_frame = not process_this_frame
        
        # Draw results
        frame = draw_face_boxes(frame, results)
        
        # Display
        cv2.imshow('Face Recognition Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()


def recognize_from_image(image_path):
    """Recognize faces in a single image."""
    recognizer = FaceRecognizer()
    
    # Load or train model
    if os.path.exists(recognizer.model_path):
        print("Loading trained model...")
        recognizer.load_model()
    else:
        print("No trained model found. Training new model...")
        recognizer.train(config.TRAIN_DIR, save_model=True)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    print(f"\nRecognizing faces in: {image_path}")
    
    # Recognize faces
    results = recognizer.recognize(image_path)
    
    # Load and draw results
    image = cv2.imread(image_path)
    image = draw_face_boxes(image, results)
    
    # Print results
    print("\nRecognition Results:")
    for i, result in enumerate(results):
        print(f"\nFace {i + 1}:")
        print(f"  Name: {result['name']}")
        print(f"  Confidence: {result['confidence']:.2%}")
    
    # Display image
    cv2.imshow('Face Recognition Result', image)
    print("\nPress any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main demo function."""
    print("=" * 60)
    print("Face Recognition Interactive Demo")
    print("=" * 60)
    print(f"Project: Face Recognition Dataset (ID: 22)")
    print(f"RSK World - https://rskworld.in/")
    print("=" * 60)
    
    print("\nSelect mode:")
    print("1. Webcam (real-time recognition)")
    print("2. Image file")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        recognize_from_webcam()
    elif choice == "2":
        image_path = input("Enter image path: ").strip()
        recognize_from_image(image_path)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()

