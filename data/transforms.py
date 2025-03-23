import random
import cv2
import numpy as np

class DualTransform:
    """Base class for transforms that modify both image and bounding boxes."""
    def __init__(self, always_apply=False, prob=1.0):
        self.always_apply = always_apply
        self.prob = prob

    def __call__(self, sample):
        if self.always_apply or random.random() < self.prob:
            return self.apply(sample)
        return sample

    def apply(self, sample):
        raise NotImplementedError

class YOLOAugmentation:
    def __init__(self, target_size=608, train=True):
        self.train = train
        self.target_size = target_size

    def normalize(self, image):
        """Normalizes image to range [0, 1]."""
        return image.astype(np.float32) / 255.0

    def resize(self, image, boxes):
        """Resizes image and updates bounding boxes."""
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.target_size, self.target_size))
        
        if isinstance(boxes, np.ndarray) and boxes.size > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (self.target_size / w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (self.target_size / h)
            
            # Ensure boxes are valid
            boxes = np.clip(boxes, 0, self.target_size)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid] if np.any(valid) else np.zeros((0, 4), dtype=np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
        
        return image, boxes

    def horizontal_flip(self, image, boxes):
        """Performs horizontal flip with box updates."""
        image = cv2.flip(image, 1)
        
        if isinstance(boxes, np.ndarray) and boxes.size > 0:
            w = image.shape[1]
            boxes = boxes.copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        
        return image, boxes

    def color_jitter(self, image):
        """Applies color jittering."""
        # Brightness
        brightness = random.uniform(0.5, 1.5)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # Contrast
        contrast = random.uniform(0.5, 1.5)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        return image

    def __call__(self, sample):
        """
        Applies the transformation pipeline to the input sample.
        
        Args:
            sample (dict): Input dictionary containing image and annotations
            
        Returns:
            dict: Transformed sample
        """
        image = sample['image']
        boxes = sample.get('boxes', np.zeros((0, 4), dtype=np.float32))
        labels = sample.get('labels', np.array([], dtype=np.int64))
        
        # Ensure initial boxes and labels are numpy arrays
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes, dtype=np.float32)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels, dtype=np.int64)
        
        if self.train:
            # Horizontal flip
            if random.random() < 0.5:
                image, boxes = self.horizontal_flip(image, boxes)
            
            # Color augmentations
            if random.random() < 0.5:
                image = self.color_jitter(image)
        
        # Always resize and normalize
        image, boxes = self.resize(image, boxes)
        image = self.normalize(image)
        
        # Final validation of outputs
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        if labels.size == 0:
            labels = np.array([], dtype=np.int64)
            
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels
        }
