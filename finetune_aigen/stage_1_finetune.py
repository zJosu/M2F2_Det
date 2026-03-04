"""
Stage 1 Fine-tuning: M2F2Det (DenseNet121) for AI-Generated Image Detection.

This script fine-tunes the M2F2Det model's DenseNet121 backbone to detect
fully AI-generated images (from diffusion models, GANs, etc.).

Uses the SAME architecture as the LLM-integrated version (llava/model/deepfake/M2F2Det/model.py)
so the output checkpoint can directly replace `M2F2_Det_densenet121.pth`.

Usage:
    python finetune_aigen/stage_1_finetune.py --data_dir ./finetune_aigen/data
    python finetune_aigen/stage_1_finetune.py --data_dir ./finetune_aigen/data --resume ./checkpoints/stage_1_aigen/best_model.pth

For Colab:
    !python finetune_aigen/stage_1_finetune.py \\
        --data_dir ./finetune_aigen/data \\
        --pretrained_ckpt ./utils/weights/M2F2_Det_densenet121.pth \\
        --output_dir ./checkpoints/stage_1_aigen \\
        --epochs 30 --batch_size 16 --lr 1e-4
"""

import os
import sys
import json
import argparse
import datetime
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, classification_report

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Dataset
# ============================================================================

class AIGenImageDataset(Dataset):
    """
    Simple image folder dataset for real vs AI-generated classification.
    
    Expected structure:
        data_dir/
            real/          -> label 0
            aigenerated/   -> label 1
    """
    
    def __init__(self, data_dir: str, transform=None, max_per_class: int = None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []  # list of (path, label)
        
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        
        for label_name, label_id in [('real', 0), ('aigenerated', 1)]:
            label_dir = os.path.join(data_dir, label_name)
            if not os.path.isdir(label_dir):
                print(f"  [WARN] Directory not found: {label_dir}")
                continue
            
            files = sorted([
                os.path.join(label_dir, f) 
                for f in os.listdir(label_dir) 
                if os.path.splitext(f)[1].lower() in img_exts
            ])
            
            if max_per_class and len(files) > max_per_class:
                files = files[:max_per_class]
            
            for f in files:
                self.samples.append((f, label_id))
        
        print(f"  Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)


def get_train_transforms():
    """Training data augmentation pipeline."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """Validation/test transform pipeline."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# Data Augmentation (matching M2F2-Det style)
# ============================================================================

class JpegCompression:
    """Random JPEG compression augmentation."""
    def __init__(self, quality_range=(30, 100), prob=0.1):
        self.quality_range = quality_range
        self.prob = prob
    
    def __call__(self, img):
        if np.random.random() < self.prob:
            import io
            quality = np.random.randint(*self.quality_range)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            img = Image.open(buffer)
        return img


class GaussianBlurAug:
    """Random Gaussian blur augmentation."""
    def __init__(self, sigma_range=(0.1, 3.0), prob=0.1):
        self.sigma_range = sigma_range
        self.prob = prob
    
    def __call__(self, img):
        if np.random.random() < self.prob:
            from PIL import ImageFilter
            sigma = np.random.uniform(*self.sigma_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


def get_train_transforms_advanced():
    """Advanced training transforms matching M2F2-Det + extra for AI-gen robustness."""
    return transforms.Compose([
        # Pre-processing augmentations (on PIL image)
        JpegCompression(quality_range=(30, 100), prob=0.15),
        GaussianBlurAug(sigma_range=(0.1, 2.0), prob=0.1),
        # Standard augmentations
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        # To tensor + normalize (ImageNet normalization for DenseNet)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# Model Loading Utilities
# ============================================================================

def load_m2f2det_model(pretrained_ckpt: str = None, device: str = 'cuda'):
    """
    Load the M2F2Det model with DenseNet121 backbone.
    Uses the LLM-integrated architecture (llava/model/deepfake/M2F2Det/model.py).
    """
    # Import the correct M2F2Det class (LLM-integrated version)
    try:
        from llava.model.deepfake.M2F2Det.model import M2F2Det
    except ImportError:
        # Fallback: try sequence version
        from sequence.models.M2F2_Det.models.model import M2F2Det
    
    print("  Building M2F2Det with DenseNet121...")
    model = M2F2Det(
        clip_text_encoder_name="openai/clip-vit-large-patch14-336",
        clip_vision_encoder_name="openai/clip-vit-large-patch14-336",
        deepfake_encoder_name='densenet121',
        hidden_size=1024,
        vision_dtype=torch.float32,
        text_dtype=torch.float32,
        deepfake_dtype=torch.float32,
        load_vision_encoder=True,
        pretrained=True,
    )
    
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        print(f"  Loading pretrained weights from: {pretrained_ckpt}")
        state_dict = torch.load(pretrained_ckpt, map_location='cpu')
        # Handle nested state dicts
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix if present (from DataParallel)
        cleaned_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                cleaned_dict[k[7:]] = v
            else:
                cleaned_dict[k] = v
        
        # Load with strict=False to handle minor architecture differences
        missing, unexpected = model.load_state_dict(cleaned_dict, strict=False)
        if missing:
            print(f"  [INFO] Missing keys: {len(missing)} (these will be randomly initialized)")
            for k in missing[:5]:
                print(f"    - {k}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")
        if unexpected:
            print(f"  [INFO] Unexpected keys: {len(unexpected)} (these will be ignored)")
    
    # Load LLaVA vision tower weights if available
    vision_tower_path = './utils/weights/vision_tower.pth'
    if os.path.exists(vision_tower_path) and model.clip_vision_encoder is not None:
        print(f"  Loading CLIP vision tower from: {vision_tower_path}")
        vt_dict = torch.load(vision_tower_path, map_location='cpu')
        vt_clean = {k.replace("vision_tower.", ""): v for k, v in vt_dict.items()}
        model.clip_vision_encoder.model.load_state_dict(vt_clean, strict=True)
    
    return model.to(device)


def freeze_clip_encoders(model):
    """Freeze CLIP vision and text encoders (they're already pretrained)."""
    frozen_params = 0
    
    if hasattr(model, 'clip_vision_encoder') and model.clip_vision_encoder is not None:
        for p in model.clip_vision_encoder.parameters():
            p.requires_grad = False
            frozen_params += p.numel()
    
    if hasattr(model, 'clip_text_encoder'):
        for p in model.clip_text_encoder.parameters():
            p.requires_grad = False
            frozen_params += p.numel()
    
    print(f"  Froze {frozen_params:,} CLIP encoder parameters")
    
    # Count trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} parameters ({100*trainable/total:.1f}%)")


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, 
                    grad_accum_steps=4, epoch=0, display_step=10):
    """Train for one epoch with gradient accumulation and mixed precision."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Epoch {epoch:3d} Train")
    
    for batch_idx, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass with mixed precision
        with autocast():
            output = model(images)
            if isinstance(output, dict):
                logits = output['pred']
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            loss = criterion(logits, labels) / grad_accum_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Metrics
        total_loss += loss.item() * grad_accum_steps
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        if (batch_idx + 1) % display_step == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = total_correct / total_samples * 100
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.1f}%'})
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / total_samples * 100
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate and compute metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(val_loader, desc="Validating"):
        images = images.to(device)
        labels = labels.to(device)
        
        with autocast():
            output = model(images)
            if isinstance(output, dict):
                logits = output['pred']
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())  # prob of being fake
    
    avg_loss = total_loss / len(val_loader)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    try:
        auc_roc = roc_auc_score(all_labels, all_probs) * 100
    except ValueError:
        auc_roc = 0.0
    try:
        avg_precision = average_precision_score(all_labels, all_probs) * 100
    except ValueError:
        avg_precision = 0.0
    f1 = f1_score(all_labels, all_preds, average='binary') * 100
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'avg_precision': avg_precision,
        'f1': f1
    }
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
        torch.save(checkpoint, best_path)


def export_deepfake_encoder_weights(model, save_path):
    """
    Export only the weights needed for the LLM-integrated version.
    This creates a checkpoint compatible with `deepfake_ckpt_path` in Stage 2/3 training.
    """
    # Save the full M2F2Det state_dict (this is what deepfake_ckpt_path loads)
    state_dict = model.state_dict()
    torch.save(state_dict, save_path)
    print(f"  [EXPORT] Deepfake encoder weights saved to: {save_path}")
    print(f"  [EXPORT] Size: {os.path.getsize(save_path) / 1e6:.1f} MB")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 1 Fine-tuning for AI-Generated Image Detection')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--pretrained_ckpt', type=str, default='./utils/weights/M2F2_Det_densenet121.pth',
                       help='Path to pretrained M2F2Det checkpoint')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/stage_1_aigen',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--grad_accum', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--freeze_clip', action='store_true', default=True,
                       help='Freeze CLIP vision and text encoders')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # ==========================================
    # 1. Load datasets
    # ==========================================
    logger.info("Loading datasets...")
    
    train_dataset = AIGenImageDataset(
        os.path.join(args.data_dir, 'train'), 
        transform=get_train_transforms_advanced()
    )
    val_dataset = AIGenImageDataset(
        os.path.join(args.data_dir, 'val'), 
        transform=get_val_transforms()
    )
    
    if len(train_dataset) == 0:
        logger.error("No training data found! Run prepare_dataset.py first.")
        return
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # ==========================================
    # 2. Load model
    # ==========================================
    logger.info("Loading M2F2Det model...")
    model = load_m2f2det_model(args.pretrained_ckpt, device=device)
    
    if args.freeze_clip:
        freeze_clip_encoders(model)
    
    # ==========================================
    # 3. Setup optimizer and scheduler
    # ==========================================
    # Use per-module learning rates (matching M2F2-Det training style)
    if hasattr(model, 'assign_lr_dict_list'):
        params_dict_list = model.assign_lr_dict_list(lr=args.lr)
        optimizer = torch.optim.AdamW(params_dict_list, weight_decay=args.weight_decay)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # ==========================================
    # 4. Resume if needed
    # ==========================================
    start_epoch = 0
    best_auc = 0.0
    
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_auc = ckpt.get('metrics', {}).get('auc_roc', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best AUC: {best_auc:.2f}%")
    
    # ==========================================
    # 5. Training loop
    # ==========================================
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum}")
    logger.info(f"Total epochs: {args.epochs}")
    logger.info("=" * 60)
    
    history = []
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            grad_accum_steps=args.grad_accum, epoch=epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.1f}%, "
            f"AUC: {val_metrics['auc_roc']:.1f}%, F1: {val_metrics['f1']:.1f}%"
        )
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })
        
        # Save checkpoint
        is_best = val_metrics['auc_roc'] > best_auc
        if is_best:
            best_auc = val_metrics['auc_roc']
            logger.info(f"  New best AUC: {best_auc:.2f}%!")
        
        if (epoch + 1) % args.save_every == 0 or is_best:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch:03d}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, save_path, is_best)
            logger.info(f"  Checkpoint saved: {save_path}")
        
        # Save history
        with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    # ==========================================
    # 6. Export final weights
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation AUC: {best_auc:.2f}%")
    
    # Load best model and export
    best_ckpt_path = os.path.join(args.output_dir, 'best_model.pth')
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])
    
    # Export weights for LLM integration
    export_path = os.path.join(args.output_dir, 'M2F2_Det_densenet121_aigen.pth')
    export_deepfake_encoder_weights(model, export_path)
    
    logger.info("\nNext steps:")
    logger.info(f"  1. Copy {export_path} to utils/weights/M2F2_Det_densenet121.pth")
    logger.info("  2. Run generate_ddvqa_data.py to create Stage 2/3 training data")
    logger.info("  3. Run Stage 2 training (projector alignment)")
    logger.info("  4. Run Stage 3 training (LoRA fine-tuning)")


if __name__ == '__main__':
    main()
