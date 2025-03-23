import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ConvBNMish

class YOLOHead(nn.Module):
    def __init__(self, num_classes, anchors, stride, in_channels):
        """
        Args:
            num_classes (int): Number of classes to detect
            anchors (list): List of anchor box pairs (w, h) for this scale
            stride (int): Stride of this detection head
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.stride = stride
        
        # Register anchors as buffer (not parameters)
        self.register_buffer('anchors', torch.tensor(anchors).float().view(-1, 2))
        
        # Output conv: (num_anchors * (5 + num_classes))
        # 5 = objectness + 4 bbox coordinates
        self.output_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * (5 + num_classes),
            kernel_size=1
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Predict
        pred = self.output_conv(x)
        
        # Reshape prediction [batch, anchors*(5+classes), height, width] -> [batch, anchors, height, width, 5+classes]
        pred = pred.view(batch_size, self.num_anchors, 5 + self.num_classes, height, width)
        pred = pred.permute(0, 1, 3, 4, 2)  # [batch, anchors, height, width, 5+classes]
        
        # Sigmoid object confidence and class probabilities
        pred_conf = torch.sigmoid(pred[..., 4:5])
        pred_cls = torch.sigmoid(pred[..., 5:])
        
        # Get bbox predictions
        pred_boxes = pred[..., :4]
        pred_xy = torch.sigmoid(pred_boxes[..., :2])  # Center coordinates
        pred_wh = torch.exp(pred_boxes[..., 2:])      # Width and height
        
        # Create grid
        grid_y, grid_x = torch.meshgrid([torch.arange(height), torch.arange(width)], indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=2).to(x.device)
        grid = grid.view(1, 1, height, width, 2).float()
        
        # Add offset to center coordinates and scale width/height
        pred_xy = (pred_xy + grid) * self.stride
        pred_wh = pred_wh * self.anchors.view(1, self.num_anchors, 1, 1, 2)
        
        # Final predictions
        pred_x1y1 = pred_xy - pred_wh / 2  # Top-left corner
        pred_x2y2 = pred_xy + pred_wh / 2  # Bottom-right corner
        pred_boxes = torch.cat([pred_x1y1, pred_x2y2], dim=-1)
        
        return pred_boxes, pred_conf, pred_cls

class YOLOLoss(nn.Module):
    def __init__(self, num_classes, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, predictions, targets, anchors):
        """
        Args:
            predictions (tuple): (pred_boxes, pred_conf, pred_cls)
            targets (dict): Contains 'boxes' and 'labels'
            anchors (tensor): Anchor boxes for this scale
        """
        pred_boxes, pred_conf, pred_cls = predictions
        batch_size = pred_boxes.size(0)
        
        # Compute IoU between predictions and targets
        ious = self._box_iou(pred_boxes, targets['boxes'])
        
        # Find best anchor for each target
        best_ious, best_anchor_idx = ious.max(dim=1)
        
        # Objectness loss
        obj_mask = torch.zeros_like(pred_conf)
        obj_mask[best_anchor_idx] = 1
        
        obj_loss = self.bce_loss(pred_conf, obj_mask)
        obj_loss = obj_loss.mean()
        
        # Only compute box loss for positive samples
        pos_mask = obj_mask > 0
        if pos_mask.sum() > 0:
            # Box coordinate loss (using CIoU loss)
            ciou_loss = self._compute_ciou(
                pred_boxes[pos_mask],
                targets['boxes'][best_anchor_idx[pos_mask]]
            )
            box_loss = ciou_loss.mean()
            
            # Class loss
            class_loss = self.bce_loss(
                pred_cls[pos_mask],
                F.one_hot(targets['labels'][best_anchor_idx[pos_mask]], self.num_classes).float()
            ).mean()
        else:
            box_loss = torch.tensor(0.).to(pred_boxes.device)
            class_loss = torch.tensor(0.).to(pred_boxes.device)
        
        # Weighted sum of losses
        total_loss = (
            self.lambda_coord * box_loss +
            obj_loss +
            self.lambda_noobj * (1 - obj_mask) * obj_loss +
            class_loss
        )
        
        return total_loss, {
            'loss': total_loss.item(),
            'box_loss': box_loss.item(),
            'obj_loss': obj_loss.item(),
            'class_loss': class_loss.item()
        }

    @staticmethod
    def _box_iou(boxes1, boxes2):
        """Compute IoU between two sets of boxes"""
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Get intersection coordinates
        inter_x1 = torch.max(boxes1[..., 0].unsqueeze(-1), boxes2[..., 0])
        inter_y1 = torch.max(boxes1[..., 1].unsqueeze(-1), boxes2[..., 1])
        inter_x2 = torch.min(boxes1[..., 2].unsqueeze(-1), boxes2[..., 2])
        inter_y2 = torch.min(boxes1[..., 3].unsqueeze(-1), boxes2[..., 3])
        
        # Calculate intersection area
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union area
        union_area = area1.unsqueeze(-1) + area2 - inter_area
        
        # Return IoU
        return inter_area / (union_area + 1e-6)

    @staticmethod
    def _compute_ciou(boxes1, boxes2):
        """Compute Complete IoU loss between two sets of boxes"""
        # IoU
        iou = YOLOLoss._box_iou(boxes1, boxes2)
        
        # Get bounding coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(-1)
        
        # Get centers
        b1_cx = (b1_x1 + b1_x2) / 2
        b1_cy = (b1_y1 + b1_y2) / 2
        b2_cx = (b2_x1 + b2_x2) / 2
        b2_cy = (b2_y1 + b2_y2) / 2
        
        # Diagonal length of the smallest enclosing box
        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)
        c_diag = (c_x2 - c_x1).pow(2) + (c_y2 - c_y1).pow(2)
        
        # Center distance
        center_dist = (b1_cx - b2_cx).pow(2) + (b1_cy - b2_cy).pow(2)
        
        # Get aspect ratios
        w1 = b1_x2 - b1_x1
        h1 = b1_y2 - b1_y1
        w2 = b2_x2 - b2_x1
        h2 = b2_y2 - b2_y1
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
        )
        
        # Compute CIoU
        alpha = v / (1 - iou + v + 1e-6)
        ciou = iou - center_dist / c_diag - alpha * v
        
        return 1 - ciou
