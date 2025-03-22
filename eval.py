import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.yolo import YOLOv4
from data.dataset import VOCDataset, VOC_CLASSES
from data.transforms import YOLOAugmentation
from configs.model_config import ModelConfig

def plot_boxes(image, boxes, labels, scores, class_names):
    """Plot bounding boxes on the image"""
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names))).tolist()
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = colors[int(label)]
        color = [int(255*c) for c in color[:3]]  # Convert to BGR
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        text = f'{class_names[int(label)]} {score:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height) = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Draw label background
        cv2.rectangle(image, (x1, y1-text_height-5), (x1+text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, text, (x1, y1-5), font, font_scale, (255, 255, 255), thickness)
    
    return image

def evaluate(checkpoint_path=None, visualization_output="outputs/visualizations"):
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(visualization_output, exist_ok=True)
    
    # Initialize dataset and dataloader
    test_transform = YOLOAugmentation(target_size=config.INPUT_WIDTH, train=False)
    test_dataset = VOCDataset(
        root_dir="data/pascal_voc",
        year="2012",
        image_set="test",  # or "val" if test set is not available
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for visualization
        shuffle=False,
        num_workers=4,
        collate_fn=VOCDataset.collate_fn,
        pin_memory=True
    )
    
    # Initialize model
    model = YOLOv4(num_classes=config.NUM_CLASSES)
    model = model.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Evaluation loop
    model.eval()
    total_detections = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Get original image for visualization
            original_image = batch['images'][0].permute(1, 2, 0).numpy()
            original_image = (original_image * 255).astype(np.uint8)
            
            # Move batch to device
            images = batch['images'].to(device)
            
            # Get predictions
            detections = model(images)
            
            # Process detections for visualization
            for sample_dets in detections:
                if len(sample_dets) > 0:
                    boxes = sample_dets[:, :4].cpu().numpy()
                    scores = sample_dets[:, 4].cpu().numpy()
                    labels = sample_dets[:, 5].cpu().numpy()
                    
                    # Filter detections by confidence
                    mask = scores > config.CONF_THRESHOLD
                    boxes = boxes[mask]
                    scores = scores[mask]
                    labels = labels[mask]
                    
                    # Plot boxes on image
                    vis_image = plot_boxes(
                        original_image.copy(),
                        boxes,
                        labels,
                        scores,
                        VOC_CLASSES
                    )
                    
                    # Save visualization
                    output_path = os.path.join(
                        visualization_output,
                        f"detection_{batch_idx:04d}.jpg"
                    )
                    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    
                    # Collect detection statistics
                    total_detections.extend([
                        {
                            'image_id': batch_idx,
                            'category_id': int(label),
                            'bbox': box.tolist(),
                            'score': float(score)
                        }
                        for box, label, score in zip(boxes, labels, scores)
                    ])
    
    # Calculate and print detection statistics
    print("\nDetection Statistics:")
    print(f"Total images processed: {len(test_loader)}")
    print(f"Total detections: {len(total_detections)}")
    
    # Calculate per-class statistics
    class_stats = {i: [] for i in range(config.NUM_CLASSES)}
    for det in total_detections:
        class_stats[det['category_id']].append(det['score'])
    
    print("\nPer-class Statistics:")
    for class_id, scores in class_stats.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"{VOC_CLASSES[class_id]}: {len(scores)} detections, "
                  f"average confidence: {avg_score:.3f}")
    
    print(f"\nResults saved to {visualization_output}")

if __name__ == "__main__":
    # Use the best model for evaluation
    evaluate(checkpoint_path="outputs/weights/best.pth")
