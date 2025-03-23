import os
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import cv2
import numpy as np

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOCDataset(Dataset):
    def __init__(self, root_dir, year="2012", image_set="train", transform=None):
        """
        Args:
            root_dir (str): Path to VOC dataset directory
            year (str): Dataset year (2007 or 2012)
            image_set (str): train/val/test
            transform (callable, optional): Transform to be applied on samples
        """
        self.root_dir = root_dir
        self.year = year
        self.image_set = image_set
        self.transform = transform
        
        # Create class name to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}
        
        # Load annotation paths
        self.image_ids = []
        set_file = os.path.join(root_dir, 'ImageSets', 'Main', f'{image_set}.txt')
        with open(set_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, 'JPEGImages', f'{img_id}.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        anno_path = os.path.join(self.root_dir, 'Annotations', f'{img_id}.xml')
        boxes, labels = self._parse_voc_xml(anno_path)
        
        sample = {
            'image': img,
            'boxes': boxes,  # N x 4 array of bbox coordinates [x_min, y_min, x_max, y_max]
            'labels': labels  # N array of class indices
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

    def _parse_voc_xml(self, anno_path):
        """Parse Pascal VOC annotation file."""
        tree = ET.parse(anno_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # Get class label
            class_name = obj.find('name').text
            if class_name not in self.class_to_idx:
                continue
            labels.append(self.class_to_idx[class_name])
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
        
        # Handle empty annotations
        if not boxes:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        
        # Ensure boxes are valid
        if boxes.size > 0:
            # Ensure boxes are within image bounds
            boxes = np.clip(boxes, 0, None)  # Prevent negative coordinates
            
            # Filter out invalid boxes
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            labels = labels[valid]
            
            if len(boxes) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.array([], dtype=np.int64)
        
        return boxes, labels

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader."""
        images = []
        boxes = []
        labels = []
        
        for sample in batch:
            images.append(torch.from_numpy(sample['image']).permute(2, 0, 1))
            # Ensure boxes and labels are tensors even when empty
            boxes.append(torch.from_numpy(sample['boxes']).float())
            labels.append(torch.from_numpy(sample['labels']).long())
        
        # Stack images (they should be same size)
        images = torch.stack(images)
        
        return {
            'images': images,
            'boxes': boxes,  # List of tensors of varying sizes
            'labels': labels  # List of tensors of varying sizes
        }
