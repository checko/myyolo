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
        total_box_loss = torch.tensor(0., device=pred_boxes.device)
        total_obj_loss = torch.tensor(0., device=pred_boxes.device)
        total_class_loss = torch.tensor(0., device=pred_boxes.device)
        num_targets = 0
        
        # Process each item in the batch
        for i in range(batch_size):
            if len(targets['boxes'][i]) == 0:  # Skip if no targets
                continue
                
            # Get targets for this batch item
            batch_boxes = targets['boxes'][i]  # tensor of shape [num_targets, 4]
            batch_labels = targets['labels'][i]  # tensor of shape [num_targets]
            
            # Compute IoU with targets
            batch_pred_boxes = pred_boxes[i]  # [num_anchors, height, width, 4]
            ious = self._box_iou(
                batch_pred_boxes.reshape(-1, 4),  # [num_anchors*height*width, 4]
                batch_boxes
            )  # [num_anchors*height*width, num_targets]
            
            # Find best anchor for each target
            best_ious, best_anchor_idx = ious.max(dim=0)  # [num_targets]
            
            # Create object mask
            obj_mask = torch.zeros_like(pred_conf[i])  # [num_anchors, height, width, 1]
            obj_mask.view(-1)[best_anchor_idx] = 1
            
            # Objectness loss
            obj_loss = self.bce_loss(pred_conf[i], obj_mask)
            total_obj_loss += obj_loss.mean()
            
            # Only compute box and class loss for positive samples
            pos_mask = obj_mask.bool().squeeze(-1)  # [num_anchors, height, width]
            if pos_mask.any():
                # Box coordinate loss (using CIoU loss)
                ciou_loss = self._compute_ciou(
                    batch_pred_boxes[pos_mask],  # [num_positive, 4]
                    batch_boxes[torch.where(pos_mask.view(-1))[0]]  # [num_positive, 4]
                )
                total_box_loss += ciou_loss.mean()
                
                # Class loss
                class_loss = self.bce_loss(
                    pred_cls[i][pos_mask],  # [num_positive, num_classes]
                    F.one_hot(batch_labels[torch.where(pos_mask.view(-1))[0]], self.num_classes).float()  # [num_positive, num_classes]
                )
                total_class_loss += class_loss.mean()
            
            num_targets += 1
        
        # Average losses over number of targets
        if num_targets > 0:
            total_box_loss /= num_targets
            total_obj_loss /= num_targets
            total_class_loss /= num_targets
        # Weighted sum of losses
        total_loss = (
            self.lambda_coord * total_box_loss +
            total_obj_loss +
            total_class_loss
        )
        
        return total_loss, {
            'loss': total_loss.item(),
            'box_loss': total_box_loss.item(),
            'obj_loss': total_obj_loss.item(),
            'class_loss': total_class_loss.item()
        }

    @staticmethod
    def _box_iou(boxes1, boxes2):
        """Compute IoU between two sets of boxes"""
        # Convert boxes to tensors if they're not already
        if not isinstance(boxes1, torch.Tensor):
            boxes1 = torch.tensor(boxes1, dtype=torch.float32)
        if not isinstance(boxes2, torch.Tensor):
            boxes2 = torch.tensor(boxes2, dtype=torch.float32)
            
        # Move tensors to the same device as boxes1
        boxes2 = boxes2.to(boxes1.device)
        
        # Compute areas
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
        # Convert boxes to tensors if they're not already
        if not isinstance(boxes1, torch.Tensor):
            boxes1 = torch.tensor(boxes1, dtype=torch.float32)
        if not isinstance(boxes2, torch.Tensor):
            boxes2 = torch.tensor(boxes2, dtype=torch.float32)
            
        # Move tensors to the same device as boxes1
        boxes2 = boxes2.to(boxes1.device)
        
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
