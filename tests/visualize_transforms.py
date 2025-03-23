import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.transforms import YOLOAugmentation

def read_voc_xml(xml_path):
    """Read Pascal VOC annotation XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    labels = []
    
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(obj.find('name').text)
    
    return np.array(boxes), np.array(labels)

def draw_boxes(image, boxes, labels):
    """Draw bounding boxes on the image."""
    image = image.copy()
    if len(boxes) > 0:
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    # Read image and annotation
    img_path = "data/pascal_voc/JPEGImages/2007_000027.jpg"
    xml_path = "data/pascal_voc/Annotations/2007_000027.xml"
    
    # Read image and annotations
    image = cv2.imread(img_path)
    boxes, labels = read_voc_xml(xml_path)
    
    # Draw original image with boxes
    original_vis = draw_boxes(image, boxes, labels)
    
    # Apply YOLOAugmentation
    transform = YOLOAugmentation(target_size=608, train=True)
    result = transform({
        'image': image,
        'boxes': boxes,
        'labels': labels
    })
    
    # Convert normalized image back to uint8 for visualization
    transformed_image = (result['image'] * 255).astype(np.uint8)
    transformed_vis = draw_boxes(transformed_image, 
                               result['boxes'], 
                               result['labels'])
    
    # Resize transformed image to match original size
    h, w = original_vis.shape[:2]
    transformed_vis_resized = cv2.resize(transformed_vis, (w, h))
    
    # Stack images horizontally
    combined = np.hstack((original_vis, transformed_vis_resized))
    
    # Save result
    cv2.imwrite('tests/transform_result.jpg', combined)

if __name__ == "__main__":
    main()
