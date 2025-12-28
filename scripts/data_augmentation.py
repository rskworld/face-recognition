"""
Data Augmentation for Face Recognition Dataset

This module provides data augmentation techniques to increase dataset diversity.

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
from typing import List, Tuple, Optional
import os


class FaceAugmenter:
    """
    Data augmentation for face images.
    """
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float = 15) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    @staticmethod
    def flip_horizontal(image: np.ndarray) -> np.ndarray:
        """
        Flip image horizontally.
        
        Args:
            image: Input image
            
        Returns:
            Horizontally flipped image
        """
        return cv2.flip(image, 1)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float = 0.2) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image
            factor: Brightness adjustment factor (-1 to 1)
            
        Returns:
            Brightness adjusted image
        """
        adjusted = image.astype(np.float32)
        adjusted = adjusted + (factor * 255)
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float = 0.2) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image
            factor: Contrast adjustment factor
            
        Returns:
            Contrast adjusted image
        """
        adjusted = image.astype(np.float32)
        mean = np.mean(adjusted)
        adjusted = (adjusted - mean) * (1 + factor) + mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted
    
    @staticmethod
    def add_noise(image: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
        """
        Add Gaussian noise to image.
        
        Args:
            image: Input image
            noise_factor: Noise intensity factor
            
        Returns:
            Noisy image
        """
        noise = np.random.normal(0, noise_factor * 255, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply Gaussian blur.
        
        Args:
            image: Input image
            kernel_size: Blur kernel size
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def crop_and_resize(image: np.ndarray, crop_factor: float = 0.9) -> np.ndarray:
        """
        Crop and resize image.
        
        Args:
            image: Input image
            crop_factor: Crop factor (0-1)
            
        Returns:
            Cropped and resized image
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        
        cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]
        resized = cv2.resize(cropped, (w, h))
        return resized
    
    def augment_image(self, image: np.ndarray, augmentations: List[str] = None) -> List[np.ndarray]:
        """
        Apply multiple augmentations to an image.
        
        Args:
            image: Input image
            augmentations: List of augmentation names to apply
            
        Returns:
            List of augmented images
        """
        if augmentations is None:
            augmentations = ['rotate', 'flip', 'brightness', 'contrast']
        
        augmented_images = []
        
        for aug_name in augmentations:
            if aug_name == 'rotate':
                for angle in [-15, 15]:
                    augmented_images.append(self.rotate(image, angle))
            elif aug_name == 'flip':
                augmented_images.append(self.flip_horizontal(image))
            elif aug_name == 'brightness':
                for factor in [-0.2, 0.2]:
                    augmented_images.append(self.adjust_brightness(image, factor))
            elif aug_name == 'contrast':
                for factor in [-0.2, 0.2]:
                    augmented_images.append(self.adjust_contrast(image, factor))
            elif aug_name == 'noise':
                augmented_images.append(self.add_noise(image))
            elif aug_name == 'blur':
                augmented_images.append(self.blur(image))
            elif aug_name == 'crop':
                augmented_images.append(self.crop_and_resize(image))
        
        return augmented_images
    
    def augment_directory(self, input_dir: str, output_dir: str, 
                       augmentations: List[str] = None, max_per_image: int = 5):
        """
        Augment all images in a directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            augmentations: List of augmentation names
            max_per_image: Maximum augmentations per image
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for person_dir in os.listdir(input_dir):
            person_path = os.path.join(input_dir, person_dir)
            if not os.path.isdir(person_path):
                continue
            
            output_person_dir = os.path.join(output_dir, person_dir)
            os.makedirs(output_person_dir, exist_ok=True)
            
            for filename in os.listdir(person_path):
                if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    image_path = os.path.join(person_path, filename)
                    image = cv2.imread(image_path)
                    
                    if image is None:
                        continue
                    
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    augmented = self.augment_image(rgb_image, augmentations)
                    
                    # Save original
                    base_name = os.path.splitext(filename)[0]
                    ext = os.path.splitext(filename)[1]
                    
                    for idx, aug_image in enumerate(augmented[:max_per_image]):
                        output_path = os.path.join(
                            output_person_dir, 
                            f"{base_name}_aug{idx}{ext}"
                        )
                        bgr_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(output_path, bgr_image)

