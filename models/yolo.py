import torch
import torch.nn as nn
from .backbone import CSPDarknet53
from .neck import YOLONeck
from .head import YOLOHead, YOLOLoss

class YOLOv4(nn.Module):
    def __init__(self, num_classes=20, pretrained_backbone=None):
        super().__init__()
        self.num_classes = num_classes
        
        # Define network components
        self.backbone = CSPDarknet53()
        self.neck = YOLONeck()
        
        # Three detection heads for different scales
        self.head_p3 = YOLOHead(
            num_classes=num_classes,
            anchors=[(12, 16), (19, 36), (40, 28)],
            stride=8,
            in_channels=256  # p3 channel dimension
        )
        self.head_p4 = YOLOHead(
            num_classes=num_classes,
            anchors=[(36, 75), (76, 55), (72, 146)],
            stride=16,
            in_channels=512  # p4 channel dimension
        )
        self.head_p5 = YOLOHead(
            num_classes=num_classes,
            anchors=[(142, 110), (192, 243), (459, 401)],
            stride=32,
            in_channels=1024  # p5 channel dimension
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Load pretrained backbone if provided
        if pretrained_backbone:
            self.backbone.load_state_dict(torch.load(pretrained_backbone))
        
        # Loss function
        self.loss_fn = YOLOLoss(num_classes)

    def forward(self, x, targets=None):
        # Backbone forward pass
        features = self.backbone(x)
        
        # Neck forward pass
        pyramid_features = self.neck(features)
        
        # Detection heads
        p3_out = self.head_p3(pyramid_features["p3"])
        p4_out = self.head_p4(pyramid_features["p4"])
        p5_out = self.head_p5(pyramid_features["p5"])
        
        if self.training and targets is not None:
            # Calculate losses for each scale
            loss_p3, loss_dict_p3 = self.loss_fn(p3_out, targets, self.head_p3.anchors)
            loss_p4, loss_dict_p4 = self.loss_fn(p4_out, targets, self.head_p4.anchors)
            loss_p5, loss_dict_p5 = self.loss_fn(p5_out, targets, self.head_p5.anchors)
            
            # Combine losses from all scales
            total_loss = loss_p3 + loss_p4 + loss_p5
            
            # Average the component losses across scales
            loss_dict = {
                'total_loss': total_loss.item(),
                'box_loss': (loss_dict_p3['box_loss'] + loss_dict_p4['box_loss'] + loss_dict_p5['box_loss']) / 3,
                'obj_loss': (loss_dict_p3['obj_loss'] + loss_dict_p4['obj_loss'] + loss_dict_p5['obj_loss']) / 3,
                'class_loss': (loss_dict_p3['class_loss'] + loss_dict_p4['class_loss'] + loss_dict_p5['class_loss']) / 3,
                'loss': total_loss.item()  # For compatibility with training loop
            }
            
            return total_loss, loss_dict
        
        # For validation/inference, return detections and dummy loss values
        if targets is None:
            detections = self._process_detections(p3_out, p4_out, p5_out)
            return detections, {'total_loss': 0.0, 'box_loss': 0.0, 'obj_loss': 0.0, 'class_loss': 0.0, 'loss': 0.0}
        else:
            # For evaluation with targets but not training
            return torch.tensor(0.0, device=x.device), {'total_loss': 0.0, 'box_loss': 0.0, 'obj_loss': 0.0, 'class_loss': 0.0, 'loss': 0.0}

    def _process_detections(self, p3_out, p4_out, p5_out):
        """Process detection outputs for inference"""
        batch_detections = []
        
        # Process each sample in the batch
        for sample_idx in range(p3_out[0].size(0)):
            detections = []
            
            # Process each scale
            for pred_boxes, pred_conf, pred_cls in [p3_out, p4_out, p5_out]:
                # Get confidence and class predictions
                conf = pred_conf[sample_idx]
                cls_pred = pred_cls[sample_idx]
                
                # Filter by confidence threshold
                conf_mask = conf > 0.1
                if not conf_mask.any():
                    continue
                
                # Get boxes, confidence scores, and class predictions
                boxes = pred_boxes[sample_idx][conf_mask]
                conf = conf[conf_mask]
                cls_pred = cls_pred[conf_mask]
                
                # Get max class probability and class index
                cls_conf, cls_idx = cls_pred.max(dim=1)
                
                # Combine confidence scores
                conf = conf.squeeze() * cls_conf
                
                # Create detection tensors
                detections.append(torch.cat([
                    boxes,
                    conf.unsqueeze(-1),
                    cls_idx.float().unsqueeze(-1)
                ], dim=1))
            
            if len(detections) > 0:
                # Combine all detections for this sample
                detections = torch.cat(detections, dim=0)
                
                # Non-maximum suppression
                keep = self._nms(detections, iou_threshold=0.45)
                detections = detections[keep]
            else:
                detections = torch.empty((0, 6)).to(p3_out[0].device)
            
            batch_detections.append(detections)
        
        return batch_detections

    @staticmethod
    def _nms(detections, iou_threshold):
        """
        Non-Maximum Suppression
        Args:
            detections: (N, 6) tensor of detections (x1, y1, x2, y2, conf, class)
            iou_threshold: IoU threshold for NMS
        Returns:
            indices of kept detections
        """
        keep = []
        
        if len(detections) == 0:
            return torch.tensor(keep)
            
        # Get coordinates
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by confidence score
        _, order = scores.sort(0, descending=True)
        
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
                
            i = order[0]
            keep.append(i)
            
            # Get IoU with remaining boxes
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= iou_threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
            
        return torch.tensor(keep)

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    # Quick model test
    model = YOLOv4(num_classes=20)
    x = torch.randn(2, 3, 608, 608)
    out = model(x)
    print("Model output for 2 samples:")
    for sample_dets in out:
        print(f"Detections shape: {sample_dets.shape}")
