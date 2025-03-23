import random
import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform

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

    def apply(self, image, **params):
        return image

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_transform_init_args_names(self):
        return ("target_size",)

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
            'boxes': combined_boxes if combined_boxes else np.zeros((0, 4)),
            'labels': np.array(combined_labels)
        }

class YOLOAugmentation:
    """
    Implements a comprehensive augmentation pipeline for YOLO training.
    Includes geometric, color, and noise transforms.
    """
    def __init__(self, target_size=608, train=True):
        """
        Args:
            target_size (int): Target size for the output image
            train (bool): Whether to use training or validation transforms
        """
        self.train = train
        self.target_size = target_size
        self.transform = self._create_transform_pipeline()

    def _create_geometric_transforms(self):
        """Creates geometric transformation pipeline."""
        return [
            Mosaic(target_size=self.target_size, prob=0.5),
            A.RandomResizedCrop(
                height=self.target_size,
                width=self.target_size,
                scale=(0.8, 1.0),
                ratio=(0.8, 1.2),
            ),
            A.HorizontalFlip(p=0.5),
        ]

    def _create_color_transforms(self):
        """Creates color transformation pipeline."""
        return [
            A.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
                p=0.5
            ),
            A.OneOf([
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(),
            ], p=0.3),
        ]

    def _create_noise_transforms(self):
        """Creates noise transformation pipeline."""
        return [
            A.OneOf([
                A.GaussNoise(),
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.2),
        ]

    def _create_transform_pipeline(self):
        """
        Creates the complete transformation pipeline based on train/val mode.
        
        Returns:
            A.Compose: Composed transformation pipeline
        """
        if not self.train:
            transforms = [
                A.Resize(height=self.target_size, width=self.target_size),
                A.Normalize(),
            ]
        else:
            transforms = (
                self._create_geometric_transforms() +
                self._create_color_transforms() +
                self._create_noise_transforms() +
                [A.Normalize()]
            )

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            )
        )

    def __call__(self, sample):
        """
        Applies the transformation pipeline to the input sample.
        
        Args:
            sample (dict): Input dictionary containing image and annotations
            
        Returns:
            dict: Transformed sample
        """
        transformed = self.transform(**{
            'image': sample['image'],
            'bboxes': sample['boxes'],
            'labels': sample['labels']
        })
        
        return {
            'image': transformed['image'],
            'boxes': np.array(transformed['bboxes']) if transformed['bboxes'] else np.zeros((0, 4)),
            'labels': np.array(transformed['labels'])
        }
