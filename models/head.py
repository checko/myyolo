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
            # Initialize empty batch losses
            batch_obj_loss = torch.zeros(1, device=pred_boxes.device)
            batch_box_loss = torch.zeros(1, device=pred_boxes.device)
            batch_class_loss = torch.zeros(1, device=pred_boxes.device)
            
            batch_pred_boxes = pred_boxes[i]  # [num_anchors, height, width, 4]
            
            # Get batch targets with empty check using numel()
            batch_boxes = targets['boxes'][i]  # tensor of shape [num_targets, 4]
            if not isinstance(batch_boxes, torch.Tensor):
                batch_boxes = torch.tensor(batch_boxes, dtype=torch.float32)
            
            # Check for empty boxes
            if batch_boxes.numel() == 0:
                # All predictions should have zero confidence
                obj_loss = self.bce_loss(pred_conf[i], torch.zeros_like(pred_conf[i]))
                batch_obj_loss = obj_loss.mean()
                total_obj_loss += batch_obj_loss
                continue
            
            # Get labels and ensure proper type
            batch_labels = targets['labels'][i]  # tensor of shape [num_targets]
            if not isinstance(batch_labels, torch.Tensor):
                batch_labels = torch.tensor(batch_labels, dtype=torch.long)

            # Move tensors to correct device
            batch_boxes = batch_boxes.to(pred_boxes.device)
            batch_labels = batch_labels.to(pred_boxes.device)
            
            # Compute IoU with targets
            ious = self._box_iou(
                batch_pred_boxes.reshape(-1, 4),  # [num_anchors*height*width, 4]
                batch_boxes
            )  # [num_anchors*height*width, num_targets]
            
            # Find best prediction for each target
            best_ious, best_anchor_idx = ious.max(dim=0)  # [num_targets]
            
            # Create object mask (using scatter_ to avoid index errors)
            obj_mask = torch.zeros(batch_pred_boxes.reshape(-1, 4).shape[0], 1, device=pred_boxes.device)
            obj_mask.scatter_(0, best_anchor_idx.unsqueeze(1), 1)
            obj_mask = obj_mask.view(pred_conf[i].shape)
            
            # Objectness loss
            obj_loss = self.bce_loss(pred_conf[i], obj_mask)
            batch_obj_loss = obj_loss.mean()
            
            # Box coordinate loss (using CIoU loss) with validation
            try:
                pred_boxes_matched = batch_pred_boxes.reshape(-1, 4)[best_anchor_idx]  # [num_targets, 4]
                target_boxes_matched = batch_boxes.clone()  # Make sure we don't modify original
                if pred_boxes_matched.size(0) == target_boxes_matched.size(0):
                    ciou_loss = self._compute_ciou(pred_boxes_matched, target_boxes_matched)
                    batch_box_loss = ciou_loss.mean()
                else:
                    print(f"Warning: Box size mismatch - pred: {pred_boxes_matched.size()}, target: {target_boxes_matched.size()}")
                    batch_box_loss = torch.zeros(1, device=pred_boxes.device)
            except Exception as e:
                print(f"Warning: Error in box loss computation: {e}")
                batch_box_loss = torch.zeros(1, device=pred_boxes.device)
            
            # Ensure labels are valid indices
            if torch.max(batch_labels) >= self.num_classes:
                print(f"Warning: Invalid class index detected: {torch.max(batch_labels)}")
                batch_labels = torch.clamp(batch_labels, 0, self.num_classes - 1)
                
            # Class loss with size check
            try:
                pred_cls_matched = pred_cls[i].reshape(-1, self.num_classes)[best_anchor_idx]
                target_cls = F.one_hot(batch_labels, self.num_classes).float()
                assert pred_cls_matched.size() == target_cls.size(), \
                    f"Size mismatch: pred_cls {pred_cls_matched.size()} vs target_cls {target_cls.size()}"
                class_loss = self.bce_loss(pred_cls_matched, target_cls)
                batch_class_loss = class_loss.mean()
            except Exception as e:
                print(f"Warning: Error in class loss computation: {e}")
                batch_class_loss = torch.tensor(0., device=pred_boxes.device)
            
            # Accumulate batch losses
            total_box_loss += batch_box_loss
            total_obj_loss += batch_obj_loss
            total_class_loss += batch_class_loss
            num_targets += 1
        
        # Average losses over batch size (not number of targets)
        total_box_loss /= batch_size
        total_obj_loss /= batch_size
        total_class_loss /= batch_size
        # Weighted sum of losses
        total_loss = (
            self.lambda_coord * total_box_loss +
            total_obj_loss +
            total_class_loss
        )
        
        # Convert to Python floats to avoid CUDA tensor memory leak
        loss_dict = {
            'total_loss': float(total_loss.item()),
            'loss': float(total_loss.item()),  # Duplicate for compatibility with training loop
            'box_loss': float(total_box_loss.item()),
            'obj_loss': float(total_obj_loss.item()),
            'class_loss': float(total_class_loss.item())
        }
        return total_loss, loss_dict

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
        
        # Get paired IoU (not pairwise)
        # Convert from [N, 4] tensors to [N, 1, 4] and [1, N, 4] for broadcasting
        boxes1_expanded = boxes1.unsqueeze(1)  # [N, 1, 4]
        boxes2_expanded = boxes2.unsqueeze(0)  # [1, N, 4]
        
        # Compute areas
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Compute intersection
        left = torch.maximum(boxes1_expanded[..., 0], boxes2_expanded[..., 0])
        top = torch.maximum(boxes1_expanded[..., 1], boxes2_expanded[..., 1])
        right = torch.minimum(boxes1_expanded[..., 2], boxes2_expanded[..., 2])
        bottom = torch.minimum(boxes1_expanded[..., 3], boxes2_expanded[..., 3])
        
        width_height = torch.clamp(right - left, min=0) * torch.clamp(bottom - top, min=0)
        union = area1.unsqueeze(-1) + area2 - width_height
        
        iou = width_height / (union + 1e-6)
        iou = torch.diagonal(iou)  # Get IoU for matched boxes only
        
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
