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
        
        # Calculate source region size to match target region aspect ratio
        target_coords = {
            0: (0, 0, center, center),  # top-left
            1: (center, 0, center * 2, center),  # top-right
            2: (0, center, center, center * 2),  # bottom-left
            3: (center, center, center * 2, center * 2)  # bottom-right
        }[quadrant]
        
        target_w = target_coords[2] - target_coords[0]
        target_h = target_coords[3] - target_coords[1]
        target_ratio = target_w / target_h
        
        if w / h > target_ratio:  # source is wider
            source_h = int(min(h, w / target_ratio))
            source_w = int(source_h * target_ratio)
        else:  # source is taller
            source_w = int(min(w, h * target_ratio))
            source_h = int(source_w / target_ratio)
        
        # Randomly select source region
        max_x = max(0, w - source_w)
        max_y = max(0, h - source_h)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        # Calculate source coordinates maintaining aspect ratio
        source_coords = (x, y, x + source_w, y + source_h)
        
        return {
            'target': target_coords,
            'source': source_coords
        }

    def _update_bboxes(self, boxes, labels, source_dims, target_coords, source_coords):
        """
        Updates bounding box coordinates for the mosaic grid.
        
        Args:
            boxes (np.array): Original bounding boxes
            labels (np.array): Original labels corresponding to boxes
            source_dims (tuple): Original image dimensions (w, h)
            target_coords (tuple): Target coordinates in mosaic (x1, y1, x2, y2)
            source_coords (tuple): Source coordinates from original image (x1, y1, x2, y2)
            
        Returns:
            tuple: (updated_boxes, updated_labels)
        """
        if len(boxes) == 0:
            return boxes, labels

        w, h = source_dims
        tx1, ty1, tx2, ty2 = target_coords
        sx1, sy1, sx2, sy2 = source_coords
        target_size = tx2 - tx1
        source_size = sx2 - sx1

        # First scale boxes from original image size to source region size
        boxes_scaled = boxes.copy()
        
        # Calculate intersection of box with source region
        intersection_x1 = np.maximum(boxes_scaled[:, 0], sx1)
        intersection_y1 = np.maximum(boxes_scaled[:, 1], sy1)
        intersection_x2 = np.minimum(boxes_scaled[:, 2], sx2)
        intersection_y2 = np.minimum(boxes_scaled[:, 3], sy2)
        
        # Calculate intersection area
        intersection_area = np.maximum(0, intersection_x2 - intersection_x1) * \
                          np.maximum(0, intersection_y2 - intersection_y1)
                          
        # Calculate original box area
        box_area = (boxes_scaled[:, 2] - boxes_scaled[:, 0]) * \
                  (boxes_scaled[:, 3] - boxes_scaled[:, 1])
                  
        # Calculate overlap ratio
        overlap_ratio = intersection_area / box_area
        
        # Filter boxes with sufficient overlap
        valid_boxes = overlap_ratio > 0.3
        if not np.any(valid_boxes):
            return np.zeros((0, 4)), np.array([])
            
        # Keep only valid boxes
        intersection_x1 = intersection_x1[valid_boxes]
        intersection_y1 = intersection_y1[valid_boxes]
        intersection_x2 = intersection_x2[valid_boxes]
        intersection_y2 = intersection_y2[valid_boxes]
        
        # Convert to source region coordinates (relative to source region top-left)
        boxes_in_source = np.zeros((np.sum(valid_boxes), 4))
        boxes_in_source[:, 0] = intersection_x1 - sx1
        boxes_in_source[:, 1] = intersection_y1 - sy1
        boxes_in_source[:, 2] = intersection_x2 - sx1
        boxes_in_source[:, 3] = intersection_y2 - sy1
        
        # Scale to target size coordinates
        scale_x = target_size / source_size
        scale_y = target_size / source_size
        boxes_scaled_target = boxes_in_source * np.array([scale_x, scale_y, scale_x, scale_y])
        
        # Add target position offset
        boxes_final = boxes_scaled_target.copy()
        boxes_final[:, [0, 2]] += tx1
        boxes_final[:, [1, 3]] += ty1
        
        return boxes_final, labels[valid_boxes]

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
                updated_boxes, updated_labels = self._update_bboxes(
                    results['boxes'],
                    results['labels'],
                    (w, h),
                    coords['target'],
                    coords['source']
                )
                combined_boxes.append(updated_boxes)
                combined_labels.extend(updated_labels)

        # Combine boxes
        if combined_boxes:
            combined_boxes = np.concatenate(combined_boxes, 0)
            
            # Calculate box properties
            box_widths = combined_boxes[:, 2] - combined_boxes[:, 0]
            box_heights = combined_boxes[:, 3] - combined_boxes[:, 1]
            box_centers_x = (combined_boxes[:, 0] + combined_boxes[:, 2]) / 2
            box_centers_y = (combined_boxes[:, 1] + combined_boxes[:, 3]) / 2
            
            # Filter boxes based on:
            # 1. Minimum size
            # 2. Center must be within mosaic
            # 3. Significant overlap with mosaic
            valid_indices = np.where(
                (box_widths > 1) &  # Width check
                (box_heights > 1) &  # Height check
                (box_centers_x >= 0) & (box_centers_x < mosaic_size) &  # X center in bounds
                (box_centers_y >= 0) & (box_centers_y < mosaic_size) &  # Y center in bounds
                (combined_boxes[:, 0] < mosaic_size) &  # Left edge check
                (combined_boxes[:, 1] < mosaic_size) &  # Top edge check
                (combined_boxes[:, 2] > 0) &  # Right edge check
                (combined_boxes[:, 3] > 0)  # Bottom edge check
            )[0]
            
            if len(valid_indices) > 0:
                combined_boxes = combined_boxes[valid_indices]
                combined_labels = np.array(combined_labels)[valid_indices]
                # Clip boxes to mosaic boundaries
                combined_boxes = np.clip(combined_boxes, 0, mosaic_size)
            else:
                combined_boxes = np.zeros((0, 4))
                combined_labels = np.array([])

        # Resize final mosaic to target size and adjust boxes
        final_mosaic = cv2.resize(mosaic_img, (self.target_size, self.target_size))
        if len(combined_boxes) > 0:
            combined_boxes = combined_boxes * (self.target_size / mosaic_size)
        
        return {
            'image': final_mosaic,
            'boxes': combined_boxes if len(combined_boxes) > 0 else np.zeros((0, 4)),
            'labels': combined_labels
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
        
        # Calculate crop size maintaining aspect ratio
        target_ratio = self.target_size / self.target_size  # 1.0
        scale_h = scale
        scale_w = scale
        
        if w / h > target_ratio:
            scale_w = scale_h * target_ratio
        else:
            scale_h = scale_w / target_ratio
            
        new_h = int(h * scale_h)
        new_w = int(w * scale_w)
        
        # Ensure valid dimensions
        new_w = min(new_w, w)
        new_h = min(new_h, h)
        
        # Random crop position
        x = random.randint(0, w - new_w)
        y = random.randint(0, h - new_h)
        
        if len(boxes) > 0:
            boxes = boxes.copy()
            
            # Calculate intersection with crop area
            intersection_x1 = np.maximum(boxes[:, 0], x)
            intersection_y1 = np.maximum(boxes[:, 1], y)
            intersection_x2 = np.minimum(boxes[:, 2], x + new_w)
            intersection_y2 = np.minimum(boxes[:, 3], y + new_h)
            
            # Calculate intersection area
            intersection_w = intersection_x2 - intersection_x1
            intersection_h = intersection_y2 - intersection_y1
            intersection_area = np.maximum(0, intersection_w) * np.maximum(0, intersection_h)
            
            # Calculate box areas
            box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
            # Calculate overlap ratios
            overlap_ratios = intersection_area / box_areas
            
            # Filter boxes by overlap ratio and valid dimensions
            valid_boxes = (overlap_ratios > 0.3) & (intersection_w > 0) & (intersection_h > 0)
            
            if np.any(valid_boxes):
                # Keep only valid boxes and labels
                valid_labels = labels[valid_boxes]
                
                # Convert coordinates relative to crop
                valid_boxes = np.stack([
                    intersection_x1[valid_boxes] - x,
                    intersection_y1[valid_boxes] - y,
                    intersection_x2[valid_boxes] - x,
                    intersection_y2[valid_boxes] - y
                ], axis=1)
                
                # Scale boxes to target size while maintaining aspect ratio
                scale = min(self.target_size / new_w, self.target_size / new_h)
                
                # Calculate padding to maintain aspect ratio
                scaled_w = new_w * scale
                scaled_h = new_h * scale
                pad_x = (self.target_size - scaled_w) / 2
                pad_y = (self.target_size - scaled_h) / 2
                
                # Scale and pad boxes
                valid_boxes = valid_boxes * scale
                valid_boxes[:, [0, 2]] += pad_x
                valid_boxes[:, [1, 3]] += pad_y
                
                boxes = valid_boxes
                labels = valid_labels
            else:
                boxes = np.zeros((0, 4))
                labels = np.array([])
        
        # Crop and resize image maintaining aspect ratio
        image = image[y:y+new_h, x:x+new_w]
        scale = min(self.target_size / new_w, self.target_size / new_h)
        scaled_size = (int(new_w * scale), int(new_h * scale))
        resized = cv2.resize(image, scaled_size)
        
        # Create padded image
        padded = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        pad_x = (self.target_size - scaled_size[0]) // 2
        pad_y = (self.target_size - scaled_size[1]) // 2
        padded[pad_y:pad_y+scaled_size[1], pad_x:pad_x+scaled_size[0]] = resized
        image = padded
        
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
                print('mosaic')
                result = Mosaic(target_size=self.target_size)(sample)
                image = result['image']
                boxes = result['boxes']
                labels = result['labels']
            
            # Random crop
            if random.random() < 0.5:
                print('crop')
                image, boxes, labels = self.random_crop(image, boxes, labels)
            
            # Horizontal flip
            if random.random() < 0.5:
                print('flip')
                image, boxes = self.horizontal_flip(image, boxes)
            
            # Color augmentations
            if random.random() < 0.5:
                print('color')
                image = self.color_jitter(image)
            
            # Add noise
            if random.random() < 0.2:
                print('noise')
                image = self.add_noise(image)
        
        # Always resize and normalize
        image, boxes = self.resize(image, boxes)
        image = self.normalize(image)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels
        }
