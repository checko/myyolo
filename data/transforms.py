import random
import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.geometric.transforms import Perspective

class Mosaic(DualTransform):
    def __init__(self, target_size=608, prob=1.0, always_apply=False):
        super().__init__(always_apply, prob)
        self.target_size = target_size

    def apply(self, image, **params):
        return image

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_transform_init_args_names(self):
        return ("target_size",)

    def __call__(self, results):
        if random.random() > self.prob:
            return results

        mosaic_img = np.zeros((self.target_size * 2, self.target_size * 2, 3), dtype=np.uint8)
        boxes = []
        labels = []
        
        # Center coordinates
        xc, yc = self.target_size, self.target_size
        
        for i in range(4):
            img = results['image']
            h, w = img.shape[:2]
            
            if i == 0:  # top-left
                x1a, y1a = 0, 0
                x2a, y2a = xc, yc
                x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
                x2b, y2b = w, h
            elif i == 1:  # top-right
                x1a, y1a = xc, 0
                x2a, y2a = xc * 2, yc
                x1b, y1b = 0, h - (y2a - y1a)
                x2b, y2b = x2a - x1a, h
            elif i == 2:  # bottom-left
                x1a, y1a = 0, yc
                x2a, y2a = xc, yc * 2
                x1b, y1b = w - (x2a - x1a), 0
                x2b, y2b = w, y2a - y1a
            else:  # bottom-right
                x1a, y1a = xc, yc
                x2a, y2a = xc * 2, yc * 2
                x1b, y1b = 0, 0
                x2b, y2b = x2a - x1a, y2a - y1a
            
            mosaic_img[y1a:y2a, x1a:x2a] = cv2.resize(
                img[y1b:y2b, x1b:x2b], (x2a - x1a, y2a - y1a)
            )
            
            padw, padh = x1a - x1b, y1a - y1b
            
            # Update bounding boxes
            if len(results['boxes']) > 0:
                boxes_i = results['boxes'].copy()
                boxes_i[:, [0, 2]] = boxes_i[:, [0, 2]] * (x2a - x1a) / w + padw
                boxes_i[:, [1, 3]] = boxes_i[:, [1, 3]] * (y2a - y1a) / h + padh
                boxes.append(boxes_i)
                labels.extend(results['labels'])
        
        # Concatenate boxes and clip to image boundaries
        if len(boxes) > 0:
            boxes = np.concatenate(boxes, 0)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.target_size * 2)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.target_size * 2)
        
        return {
            'image': cv2.resize(mosaic_img, (self.target_size, self.target_size)),
            'boxes': boxes,
            'labels': np.array(labels)
        }

class YOLOAugmentation:
    def __init__(self, target_size=608, train=True):
        self.train = train
        self.target_size = target_size
        
        if train:
            self.transform = A.Compose([
                Mosaic(target_size=target_size, prob=0.5),
                A.RandomResizedCrop(
                    height=target_size,
                    width=target_size,
                    scale=(0.8, 1.0),
                    ratio=(0.8, 1.2),
                ),
                A.HorizontalFlip(p=0.5),
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
                A.OneOf([
                    A.GaussNoise(),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                ], p=0.2),
                A.Normalize(),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            ))
        else:
            self.transform = A.Compose([
                A.Resize(height=target_size, width=target_size),
                A.Normalize(),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            ))
    
    def __call__(self, sample):
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
