import sys
import os
import numpy as np
import cv2
import unittest

# Add parent directory to path to import transforms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.transforms import Mosaic, YOLOAugmentation

class TestTransforms(unittest.TestCase):
    def setUp(self):
        # Create a sample 100x100 image with a red square
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.image, (30, 30), (70, 70), (0, 0, 255), -1)
        
        # Sample bounding box for the red square [x1, y1, x2, y2]
        self.boxes = np.array([[30, 30, 70, 70]], dtype=np.float32)
        
        # Sample label
        self.labels = np.array([0], dtype=np.int64)
        
        # Sample data dictionary
        self.sample = {
            'image': self.image,
            'boxes': self.boxes,
            'labels': self.labels
        }

    def test_mosaic(self):
        """Test Mosaic transformation"""
        mosaic = Mosaic(target_size=100)
        result = mosaic(self.sample)

        # Check output types and shapes
        self.assertIsInstance(result['image'], np.ndarray)
        self.assertEqual(result['image'].shape, (100, 100, 3))
        self.assertIsInstance(result['boxes'], np.ndarray)
        self.assertIsInstance(result['labels'], np.ndarray)

        # Save result for visualization
        cv2.imwrite('tests/mosaic_result.jpg', result['image'])
        
        print("\nMosaic Transform Results:")
        print(f"Image shape: {result['image'].shape}")
        print(f"Boxes shape: {result['boxes'].shape}")
        print(f"Labels shape: {result['labels'].shape}")
        print(f"Boxes: \n{result['boxes']}")

    def test_yolo_augmentation(self):
        """Test YOLOAugmentation pipeline"""
        aug = YOLOAugmentation(target_size=100)
        result = aug(self.sample)

        # Check output types and shapes
        self.assertIsInstance(result['image'], np.ndarray)
        self.assertEqual(result['image'].shape, (100, 100, 3))
        self.assertIsInstance(result['boxes'], np.ndarray)
        self.assertIsInstance(result['labels'], np.ndarray)

        # Save result for visualization
        # Convert normalized image back to uint8 for saving
        img_uint8 = (result['image'] * 255).astype(np.uint8)
        cv2.imwrite('tests/yolo_aug_result.jpg', img_uint8)
        
        print("\nYOLO Augmentation Results:")
        print(f"Image shape: {result['image'].shape}")
        print(f"Boxes shape: {result['boxes'].shape}")
        print(f"Labels shape: {result['labels'].shape}")
        print(f"Normalized image range: [{result['image'].min():.3f}, {result['image'].max():.3f}]")
        print(f"Boxes: \n{result['boxes']}")

if __name__ == '__main__':
    # Create tests directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    unittest.main(argv=[''], verbosity=2)
