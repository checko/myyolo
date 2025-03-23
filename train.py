import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time

from models.yolo import YOLOv4
from data.dataset import VOCDataset
from data.transforms import YOLOAugmentation
from configs.model_config import ModelConfig

def train():
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs("outputs/weights", exist_ok=True)
    os.makedirs("outputs/tensorboard", exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter("outputs/tensorboard")
    
    # Initialize datasets and dataloaders
    train_transform = YOLOAugmentation(target_size=config.INPUT_WIDTH, train=True)
    val_transform = YOLOAugmentation(target_size=config.INPUT_WIDTH, train=False)
    
    train_dataset = VOCDataset(
        root_dir="data/pascal_voc",
        year="2012",
        image_set="train",
        transform=train_transform
    )
    
    val_dataset = VOCDataset(
        root_dir="data/pascal_voc",
        year="2012",
        image_set="val",
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=VOCDataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=VOCDataset.collate_fn,
        pin_memory=True
    )
    
    # Initialize model
    model = YOLOv4(num_classes=config.NUM_CLASSES)
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.MAX_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=config.WARMUP_EPOCHS/config.MAX_EPOCHS
    )
    
    # Training loop
    best_map = 0
    for epoch in range(config.MAX_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.MAX_EPOCHS}")
        
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_box_loss = 0
        epoch_obj_loss = 0
        epoch_class_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch['images'].to(device)
            targets = {
                'boxes': [b.to(device) for b in batch['boxes']],
                'labels': [l.to(device) for l in batch['labels']]
            }
            
            # Forward pass
            loss, loss_dict = model(images, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # Get loss values (already converted to float in loss_dict)
            loss_float = loss.item()
            
            # Update metrics (loss_dict values are already floats)
            epoch_loss += loss_float
            epoch_box_loss += loss_dict['box_loss']
            epoch_obj_loss += loss_dict['obj_loss']
            epoch_class_loss += loss_dict['class_loss']
            
            # Update progress bar with current loss
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })
            
            # Log to tensorboard (every 100 iterations)
            if batch_idx % 100 == 0:
                iteration = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train_total', loss_dict['total_loss'], iteration)
                writer.add_scalar('Loss/train_box', epoch_box_loss/(batch_idx+1), iteration)
                writer.add_scalar('Loss/train_obj', epoch_obj_loss/(batch_idx+1), iteration)
                writer.add_scalar('Loss/train_class', epoch_class_loss/(batch_idx+1), iteration)
                writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], iteration)
        
        # Calculate epoch metrics
        epoch_loss /= len(train_loader)
        epoch_box_loss /= len(train_loader)
        epoch_obj_loss /= len(train_loader)
        epoch_class_loss /= len(train_loader)
        
        # Log epoch metrics
        writer.add_scalar('Epoch/train_loss', epoch_loss, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['images'].to(device)
                targets = {
                    'boxes': [b.to(device) for b in batch['boxes']],
                    'labels': [l.to(device) for l in batch['labels']]
                }
                
                loss, loss_dict = model(images, targets)
                val_loss += loss_dict['total_loss']  # Use total_loss for consistency
        
        val_loss /= len(val_loader)
        writer.add_scalar('Epoch/val_loss', val_loss, epoch)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss
        }
        
        # Save best model
        if val_loss < best_map:
            best_map = val_loss
            torch.save(checkpoint, f"outputs/weights/best.pth")
        
        # Save latest model
        torch.save(checkpoint, f"outputs/weights/latest.pth")
        
        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    train()
