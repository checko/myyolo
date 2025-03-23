import random
import cv2
import numpy as np

class DualTransform:
    """Base class for transforms that modify both image and bounding boxes."""
    def __init__(self, always_apply=False, prob=1.0):
        self.always_apply = always_apply
        self.prob = prob

    def __call__(self, *args, **kwargs):
        if self.always_apply or random.random() < self.prob:
            return self.apply(*args, **kwargs)
        return kwargs

    def apply(self, **params):
        raise NotImplementedError

class Mosaic(DualTransform):
    """
    Implements mosaic augmentation by combining 4 images into one.
    This creates a richer context for training object detection models.
    """
    def __init__(self, target_size=608, prob=1.0, always_apply=False):
        """
        Args:
            target_size (int): Final size of the mosaic image
            prob (float): Probability of applying the transform
            always_apply (bool): Whether to always apply the transform
        """
        super().__init__(always_apply, prob)
        self.target_size = target_size

    def _create_mosaic_grid(self, image, quadrant):
        """
        Creates coordinates for placing an image in a specific quadrant of the mosaic.
        
        Args:
            image (np.array): Input image
            quadrant (int): Quadrant number (0-3, starting from top-left clockwise)
            
        Returns:
            tuple: Source and target coordinates for the image placement
        """
        h, w = image.shape[:2]
        center = self.target_size
        
        # Define quadrant coordinates
        coords = {
            0: {'target': (0, 0, center, center),  # top-left
                'source': (w - center, h - center, w, h)},
            1: {'target': (center, 0, center * 2, center),  # top-right
                'source': (0, h - center, center, h)},
            2: {'target': (0, center, center, center * 2),  # bottom-left
                'source': (w - center, 0, w, center)},
            3: {'target': (center, center, center * 2, center * 2),  # bottom-right
                'source': (0, 0, center, center)}
        }
        
        return coords[quadrant]

    def _update_bboxes(self, boxes, source_dims, target_coords):
        """
        Updates bounding box coordinates for the mosaic grid.
        
        Args:
            boxes (np.array): Original bounding boxes
            source_dims (tuple): Original image dimensions (w, h)
            target_coords (tuple): Target coordinates in mosaic (x1, y1, x2, y2)
            
        Returns:
            np.array: Updated bounding box coordinates
        """
        if len(boxes) == 0:
            return boxes

        w, h = source_dims
        tx1, ty1, _, _ = target_coords
        tw = th = self.target_size
        
        # Scale and translate boxes
        boxes_scaled = boxes.copy()
        boxes_scaled[:, [0, 2]] = boxes_scaled[:, [0, 2]] * (tw / w) + tx1
        boxes_scaled[:, [1, 3]] = boxes_scaled[:, [1, 3]] * (th / h) + ty1
        
        return boxes_scaled

    def __call__(self, results):
        """
        Applies mosaic augmentation to the input image and bounding boxes.
        
        Args:
            results (dict): Input dictionary containing image and annotations
            
        Returns:
            dict: Augmented image and updated annotations
        """
        if random.random() > self.prob:
            return results

        # Initialize mosaic image
        mosaic_size = self.target_size * 2
        mosaic_img = np.zeros((mosaic_size, mosaic_size, 3), dtype=np.uint8)
        combined_boxes = []
        combined_labels = []

        # Process each quadrant
        for quadrant in range(4):
            image = results['image']
            h, w = image.shape[:2]
            
            # Get coordinates for current quadrant
            coords = self._create_mosaic_grid(image, quadrant)
            tx1, ty1, tx2, ty2 = coords['target']
            sx1, sy1, sx2, sy2 = coords['source']
            
            # Place image in mosaic
            mosaic_img[ty1:ty2, tx1:tx2] = cv2.resize(
                image[sy1:sy2, sx1:sx2], 
                (tx2 - tx1, ty2 - ty1)
            )
            
            # Update bounding boxes
            if len(results['boxes']) > 0:
                updated_boxes = self._update_bboxes(
                    results['boxes'],
                    (w, h),
                    coords['target']
                )
                combined_boxes.append(updated_boxes)
                combined_labels.extend(results['labels'])

        # Combine and clip boxes
        if combined_boxes:
            combined_boxes = np.concatenate(combined_boxes, 0)
            combined_boxes = np.clip(
                combined_boxes,
                0,
                mosaic_size
            )

        # Resize final mosaic to target size
        final_mosaic = cv2.resize(mosaic_img, (self.target_size, self.target_size))
        
        return {
            'image': final_mosaic,
            'boxes': combined_boxes if len(combined_boxes) > 0 else np.zeros((0, 4)),
            'labels': np.array(combined_labels)
        }

class YOLOAugmentation:
    """
    Implements a comprehensive augmentation pipeline for YOLO training.
    Includes geometric, color, and noise transforms using OpenCV.
    """
    def __init__(self, target_size=608, train=True):
        """
        Args:
            target_size (int): Target size for the output image
            train (bool): Whether to use training or validation transforms
        """
        self.train = train
        self.target_size = target_size

    def normalize(self, image):
        """Normalizes image to range [0, 1]."""
        return image.astype(np.float32) / 255.0

    def resize(self, image, boxes):
        """Resizes image and updates bounding boxes."""
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.target_size, self.target_size))
        
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (self.target_size / w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (self.target_size / h)
        
        return image, boxes

    def random_crop(self, image, boxes, labels):
        """Performs random crop with box updates."""
        h, w = image.shape[:2]
        min_scale = 0.8
        scale = random.uniform(min_scale, 1.0)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        x = random.randint(0, w - new_w)
        y = random.randint(0, h - new_h)
        
        image = image[y:y+new_h, x:x+new_w]
        
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] - x, 0, new_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] - y, 0, new_h)
            
            # Remove boxes that are too small or outside crop
            valid_boxes = []
            valid_labels = []
            
            for box, label in zip(boxes, labels):
                if (box[2] - box[0]) > 1 and (box[3] - box[1]) > 1:
                    valid_boxes.append(box)
                    valid_labels.append(label)
            
            boxes = np.array(valid_boxes) if valid_boxes else np.zeros((0, 4))
            labels = np.array(valid_labels)
        
        return image, boxes, labels

    def horizontal_flip(self, image, boxes):
        """Performs horizontal flip with box updates."""
        image = cv2.flip(image, 1)
        
        if len(boxes) > 0:
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
        
        # Saturation
        saturation = random.uniform(0.5, 1.5)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, -1)
        image = np.clip(image * saturation + gray * (1 - saturation), 0, 255).astype(np.uint8)
        
        return image

    def add_noise(self, image):
        """Adds Gaussian noise to image."""
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
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
        boxes = sample['boxes']
        labels = sample['labels']

        if self.train:
            # Apply Mosaic augmentation with 50% probability
            if random.random() < 0.5:
                result = Mosaic(target_size=self.target_size)(sample)
                image = result['image']
                boxes = result['boxes']
                labels = result['labels']
            
            # Random crop
            if random.random() < 0.5:
                image, boxes, labels = self.random_crop(image, boxes, labels)
            
            # Horizontal flip
            if random.random() < 0.5:
                image, boxes = self.horizontal_flip(image, boxes)
            
            # Color augmentations
            if random.random() < 0.5:
                image = self.color_jitter(image)
            
            # Add noise
            if random.random() < 0.2:
                image = self.add_noise(image)
        
        # Always resize and normalize
        image, boxes = self.resize(image, boxes)
        image = self.normalize(image)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels
        }
