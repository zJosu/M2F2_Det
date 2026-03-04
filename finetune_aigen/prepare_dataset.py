"""
Dataset Preparation Script for AI-Generated Image Detection Fine-tuning.

This script downloads and organizes datasets of real and AI-generated face images
for fine-tuning the M2F2-Det model.

Supported data sources:
  1. 140k Real and Fake Faces (Kaggle) - FFHQ real + StyleGAN generated
  2. HuggingFace datasets for diffusion-model-generated images
  3. Manual image folder for user-provided images (e.g., Grok-generated)

Usage:
    python prepare_dataset.py --output_dir ./data --source kaggle
    python prepare_dataset.py --output_dir ./data --source huggingface
    python prepare_dataset.py --output_dir ./data --source local --local_real /path/to/real --local_fake /path/to/fake

For Colab:
    !python finetune_aigen/prepare_dataset.py --output_dir ./finetune_aigen/data --source kaggle
"""

import os
import sys
import json
import shutil
import random
import argparse
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from tqdm import tqdm


def create_directory_structure(output_dir: str):
    """Create the train/val/test directory structure."""
    for split in ['train', 'val', 'test']:
        for label in ['real', 'aigenerated']:
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)
    print(f"[OK] Directory structure created at {output_dir}")


def resize_and_save(img_path: str, save_path: str, size: int = 256):
    """Resize image to target size and save."""
    try:
        img = Image.open(img_path).convert('RGB')
        # Resize to square (the model will crop to 224x224 during training)
        img = img.resize((size, size), Image.LANCZOS)
        img.save(save_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"  [WARN] Failed to process {img_path}: {e}")
        return False


def split_files(file_list: list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split file list into train/val/test sets."""
    random.seed(seed)
    random.shuffle(file_list)
    n = len(file_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        'train': file_list[:n_train],
        'val': file_list[n_train:n_train + n_val],
        'test': file_list[n_train + n_val:]
    }


def prepare_from_kaggle_140k(output_dir: str, max_per_class: int = 10000):
    """
    Download and prepare the '140k Real and Fake Faces' dataset from Kaggle.
    
    This dataset contains:
    - 70K real faces from FFHQ
    - 70K fake faces from StyleGAN
    
    Requires: pip install kaggle
    Kaggle API credentials must be set up (~/.kaggle/kaggle.json)
    """
    print("\n=== Preparing 140k Real and Fake Faces (Kaggle) ===")
    
    try:
        import kaggle
    except ImportError:
        print("  Installing kaggle package...")
        os.system(f"{sys.executable} -m pip install kaggle")
        import kaggle
    
    download_dir = os.path.join(output_dir, '_downloads', 'kaggle_140k')
    os.makedirs(download_dir, exist_ok=True)
    
    # Download dataset
    print("  Downloading dataset from Kaggle...")
    print("  (Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables or ~/.kaggle/kaggle.json)")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('xhlulu/140k-real-and-fake-faces', path=download_dir, unzip=True)
    except Exception as e:
        print(f"  [ERROR] Kaggle download failed: {e}")
        print("  Please download manually from: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
        print(f"  Extract to: {download_dir}")
        return False
    
    # Find the extracted folders
    real_dir = None
    fake_dir = None
    for root, dirs, files in os.walk(download_dir):
        if 'real' in [d.lower() for d in dirs]:
            real_dir = os.path.join(root, 'real')
        if 'fake' in [d.lower() for d in dirs]:
            fake_dir = os.path.join(root, 'fake')
    
    if real_dir is None or fake_dir is None:
        # Try alternate structure
        for root, dirs, files in os.walk(download_dir):
            jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(jpg_files) > 1000:
                if 'real' in root.lower():
                    real_dir = root
                elif 'fake' in root.lower():
                    fake_dir = root
    
    if real_dir is None or fake_dir is None:
        print(f"  [ERROR] Could not find real/fake folders in {download_dir}")
        print(f"  Available contents: {os.listdir(download_dir)}")
        return False
    
    return _organize_folder_dataset(output_dir, real_dir, fake_dir, max_per_class, 'kaggle')


def prepare_from_huggingface(output_dir: str, max_per_class: int = 10000):
    """
    Download and prepare AI-generated face datasets from HuggingFace.
    
    Uses multiple small datasets for diversity:
    - OpenRL/DeepFakeFace (SD-generated faces)
    - or similar available datasets
    """
    print("\n=== Preparing Dataset from HuggingFace ===")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("  Installing datasets package...")
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset
    
    real_images = []
    fake_images = []
    
    # Try to load a dataset with both real and fake faces
    dataset_options = [
        ("Nithishma/ai_vs_real_image_detection", None),
        ("date3k2/ai-generated-real-face", None),
    ]
    
    for dataset_name, config in dataset_options:
        print(f"  Trying to load: {dataset_name}...")
        try:
            if config:
                ds = load_dataset(dataset_name, config, split='train')
            else:
                ds = load_dataset(dataset_name, split='train')
            
            print(f"  Loaded {len(ds)} samples. Columns: {ds.column_names}")
            
            # Save images to temp dir
            temp_real = os.path.join(output_dir, '_downloads', 'hf_real')
            temp_fake = os.path.join(output_dir, '_downloads', 'hf_fake')
            os.makedirs(temp_real, exist_ok=True)
            os.makedirs(temp_fake, exist_ok=True)
            
            # Handle different column formats
            label_col = None
            image_col = None
            for col in ds.column_names:
                if col.lower() in ['label', 'labels', 'class', 'target']:
                    label_col = col
                if col.lower() in ['image', 'img', 'pixel_values']:
                    image_col = col
            
            if label_col and image_col:
                for i, sample in enumerate(tqdm(ds, desc="Processing HF data")):
                    if i >= max_per_class * 2:
                        break
                    label = sample[label_col]
                    img = sample[image_col]
                    if isinstance(img, Image.Image):
                        if label == 0 and len(real_images) < max_per_class:
                            path = os.path.join(temp_real, f"hf_{i:06d}.jpg")
                            img.save(path, 'JPEG', quality=95)
                            real_images.append(path)
                        elif label == 1 and len(fake_images) < max_per_class:
                            path = os.path.join(temp_fake, f"hf_{i:06d}.jpg")
                            img.save(path, 'JPEG', quality=95)
                            fake_images.append(path)
                
                if real_images and fake_images:
                    print(f"  Got {len(real_images)} real, {len(fake_images)} fake from {dataset_name}")
                    return _organize_folder_dataset(output_dir, temp_real, temp_fake, max_per_class, 'hf')
        except Exception as e:
            print(f"  [WARN] Failed to load {dataset_name}: {e}")
            continue
    
    print("  [ERROR] No suitable HuggingFace dataset found.")
    print("  Please use --source local with manually downloaded images.")
    return False


def prepare_from_local(output_dir: str, local_real: str, local_fake: str, max_per_class: int = 10000):
    """
    Prepare dataset from local directories of real and fake images.
    
    Args:
        output_dir: Output directory for organized dataset
        local_real: Path to directory containing real face images
        local_fake: Path to directory containing AI-generated face images
        max_per_class: Maximum number of images per class
    """
    print("\n=== Preparing Dataset from Local Directories ===")
    
    if not os.path.isdir(local_real):
        print(f"  [ERROR] Real image directory not found: {local_real}")
        return False
    if not os.path.isdir(local_fake):
        print(f"  [ERROR] Fake image directory not found: {local_fake}")
        return False
    
    return _organize_folder_dataset(output_dir, local_real, local_fake, max_per_class, 'local')


def prepare_synthetic_test(output_dir: str, num_per_class: int = 100):
    """
    Create a small synthetic test dataset using random noise and transforms
    (for pipeline validation only; not for real evaluation).
    
    This is useful for testing the training pipeline without downloading large datasets.
    """
    import numpy as np
    
    print("\n=== Creating Synthetic Test Dataset (for pipeline validation only) ===")
    
    create_directory_structure(output_dir)
    
    for split in ['train', 'val', 'test']:
        n = num_per_class if split == 'train' else num_per_class // 5
        
        for i in tqdm(range(n), desc=f"  Generating {split}/real"):
            # Create a "real-looking" placeholder (random face-colored image)
            img = np.random.randint(120, 220, size=(256, 256, 3), dtype=np.uint8)
            # Add some natural noise patterns
            img = Image.fromarray(img)
            path = os.path.join(output_dir, split, 'real', f'synth_real_{i:06d}.jpg')
            img.save(path, 'JPEG', quality=90)
        
        for i in tqdm(range(n), desc=f"  Generating {split}/aigenerated"):
            # Create a "AI-generated-looking" placeholder (smoother, more uniform)
            img = np.random.randint(140, 200, size=(256, 256, 3), dtype=np.uint8)
            # Make it smoother (AI images tend to be smoother)
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, sigma=2)
            img = Image.fromarray(np.uint8(img))
            path = os.path.join(output_dir, split, 'aigenerated', f'synth_fake_{i:06d}.jpg')
            img.save(path, 'JPEG', quality=95)
    
    _print_dataset_stats(output_dir)
    return True


def _organize_folder_dataset(output_dir: str, real_dir: str, fake_dir: str, 
                             max_per_class: int, source_tag: str):
    """Organize images from real and fake directories into train/val/test splits."""
    
    create_directory_structure(output_dir)
    
    # Collect all image paths
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    real_files = sorted([
        os.path.join(real_dir, f) for f in os.listdir(real_dir) 
        if os.path.splitext(f)[1].lower() in img_exts
    ])[:max_per_class]
    
    fake_files = sorted([
        os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
        if os.path.splitext(f)[1].lower() in img_exts
    ])[:max_per_class]
    
    print(f"  Found {len(real_files)} real images, {len(fake_files)} fake images")
    
    # Split
    real_splits = split_files(real_files)
    fake_splits = split_files(fake_files)
    
    # Copy and resize
    for split in ['train', 'val', 'test']:
        print(f"\n  Processing {split} split...")
        for src_path in tqdm(real_splits[split], desc=f"    Real ({split})"):
            fname = f"{source_tag}_real_{Path(src_path).stem}.jpg"
            dst_path = os.path.join(output_dir, split, 'real', fname)
            resize_and_save(src_path, dst_path, size=256)
        
        for src_path in tqdm(fake_splits[split], desc=f"    Fake ({split})"):
            fname = f"{source_tag}_fake_{Path(src_path).stem}.jpg"
            dst_path = os.path.join(output_dir, split, 'aigenerated', fname)
            resize_and_save(src_path, dst_path, size=256)
    
    _print_dataset_stats(output_dir)
    return True


def _print_dataset_stats(output_dir: str):
    """Print dataset statistics."""
    print("\n=== Dataset Statistics ===")
    total = 0
    for split in ['train', 'val', 'test']:
        for label in ['real', 'aigenerated']:
            d = os.path.join(output_dir, split, label)
            if os.path.isdir(d):
                count = len([f for f in os.listdir(d) if not f.startswith('.')])
                print(f"  {split}/{label}: {count} images")
                total += count
    print(f"  Total: {total} images")
    
    # Save metadata
    metadata = {
        'total_images': total,
        'splits': {}
    }
    for split in ['train', 'val', 'test']:
        metadata['splits'][split] = {}
        for label in ['real', 'aigenerated']:
            d = os.path.join(output_dir, split, label)
            if os.path.isdir(d):
                metadata['splits'][split][label] = len([f for f in os.listdir(d) if not f.startswith('.')])
    
    with open(os.path.join(output_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {os.path.join(output_dir, 'dataset_metadata.json')}")


def add_more_images(output_dir: str, new_images_dir: str, label: str, source_tag: str = 'extra'):
    """
    Add more images to an existing dataset.
    
    Args:
        output_dir: Root dataset directory
        new_images_dir: Directory with new images
        label: 'real' or 'aigenerated'
        source_tag: Prefix for filenames
    """
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    files = sorted([
        os.path.join(new_images_dir, f) for f in os.listdir(new_images_dir)
        if os.path.splitext(f)[1].lower() in img_exts
    ])
    
    splits = split_files(files)
    
    for split in ['train', 'val', 'test']:
        dest_dir = os.path.join(output_dir, split, label)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Find next available index
        existing = os.listdir(dest_dir)
        idx_start = len(existing)
        
        for i, src_path in enumerate(tqdm(splits[split], desc=f"  Adding to {split}/{label}")):
            fname = f"{source_tag}_{idx_start + i:06d}.jpg"
            dst_path = os.path.join(dest_dir, fname)
            resize_and_save(src_path, dst_path, size=256)
    
    _print_dataset_stats(output_dir)


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for AI-generated image detection fine-tuning')
    parser.add_argument('--output_dir', type=str, default='./finetune_aigen/data',
                       help='Output directory for the dataset')
    parser.add_argument('--source', type=str, choices=['kaggle', 'huggingface', 'local', 'synthetic'],
                       default='synthetic',
                       help='Data source to use')
    parser.add_argument('--max_per_class', type=int, default=10000,
                       help='Maximum number of images per class')
    parser.add_argument('--local_real', type=str, default=None,
                       help='Path to local real face images directory')
    parser.add_argument('--local_fake', type=str, default=None,
                       help='Path to local AI-generated face images directory')
    
    # For adding extra images to existing dataset
    parser.add_argument('--add_images', type=str, default=None,
                       help='Path to additional images to add')
    parser.add_argument('--add_label', type=str, choices=['real', 'aigenerated'], default=None,
                       help='Label for additional images')
    parser.add_argument('--add_tag', type=str, default='extra',
                       help='Source tag for additional images')
    
    args = parser.parse_args()
    
    # Add more images to existing dataset
    if args.add_images:
        if args.add_label is None:
            print("[ERROR] --add_label is required when using --add_images")
            return
        add_more_images(args.output_dir, args.add_images, args.add_label, args.add_tag)
        return
    
    # Prepare new dataset
    if args.source == 'kaggle':
        success = prepare_from_kaggle_140k(args.output_dir, args.max_per_class)
    elif args.source == 'huggingface':
        success = prepare_from_huggingface(args.output_dir, args.max_per_class)
    elif args.source == 'local':
        if args.local_real is None or args.local_fake is None:
            print("[ERROR] --local_real and --local_fake are required for local source")
            return
        success = prepare_from_local(args.output_dir, args.local_real, args.local_fake, args.max_per_class)
    elif args.source == 'synthetic':
        success = prepare_synthetic_test(args.output_dir, num_per_class=500)
    
    if success:
        print("\n[DONE] Dataset preparation complete!")
        print(f"  Dataset saved to: {args.output_dir}")
        print("\n  Next step: Run Stage 1 fine-tuning:")
        print(f"  python finetune_aigen/stage_1_finetune.py --data_dir {args.output_dir}")
    else:
        print("\n[FAILED] Dataset preparation failed. See errors above.")


if __name__ == '__main__':
    main()
