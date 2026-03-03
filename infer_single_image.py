#!/usr/bin/env python3
"""
M2F2-Det Stage 1: Single Image Deepfake Detection (CPU compatible)

Usage:
    python infer_single_image.py --image path/to/face_image.jpg
    python infer_single_image.py --image path/to/face_image.jpg --weights checkpoints/stage_1/current_model_180.pth

Requirements:
    - Pre-trained CLIP vision tower weights in utils/weights/vision_tower.pth
    - Stage 1 detector weights in checkpoints/stage_1/current_model_180.pth
    - CLIP model will be auto-downloaded from HuggingFace on first run
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sequence.models.M2F2_Det.models.model import M2F2Det


def load_model(weights_path, vision_tower_path, device):
    """Load M2F2Det model with weights."""
    print("[1/4] Building M2F2Det model...")
    model = M2F2Det(
        clip_text_encoder_name="openai/clip-vit-large-patch14-336",
        clip_vision_encoder_name="openai/clip-vit-large-patch14-336",
        deepfake_encoder_name='efficientnet_b4',
        hidden_size=1792,
    )

    # Load LLaVA vision tower weights
    print("[2/4] Loading CLIP vision tower weights...")
    if not os.path.exists(vision_tower_path):
        print(f"  ERROR: Vision tower weights not found at: {vision_tower_path}")
        print(f"  Download from: https://drive.google.com/file/d/19oEpKB96xJVSrwkLV0ewje-W2dfBAR58/view")
        print(f"  Place in: utils/weights/vision_tower.pth")
        sys.exit(1)

    llava_vision_tower = torch.load(vision_tower_path, map_location=device, weights_only=True)
    vision_tower_dict = {}
    for k, v in llava_vision_tower.items():
        vision_tower_dict[k.replace("vision_tower.", "")] = v
    if model.clip_vision_encoder is not None:
        model.clip_vision_encoder.model.load_state_dict(vision_tower_dict, strict=True)
        print("  CLIP vision tower loaded OK.")

    # Wrap in DataParallel (needed because checkpoint was saved with DataParallel)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Load detector weights
    print("[3/4] Loading detector weights...")
    if not os.path.exists(weights_path):
        print(f"  ERROR: Detector weights not found at: {weights_path}")
        print(f"  Download from: https://drive.google.com/file/d/1X1ZUZkCwqg9mrsqoOS0EoO3v5WABNBAw/view")
        print(f"  Place in: checkpoints/stage_1/current_model_180.pth")
        sys.exit(1)

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print("  Detector weights loaded OK.")

    model.eval()
    return model


def preprocess_image(image_path):
    """Load and preprocess a single image to tensor [0, 1] range."""
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert('RGB')
    # Resize to 224x224 as expected by the model
    image = image.resize((224, 224), Image.BILINEAR)
    # Convert to tensor in [0, 1] range
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # HWC -> CHW
    return image_tensor.unsqueeze(0)  # Add batch dimension


def main():
    parser = argparse.ArgumentParser(description="M2F2-Det: Single Image Deepfake Detection")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input face image (jpg/png)")
    parser.add_argument("--weights", type=str,
                        default="./checkpoints/stage_1/current_model_180.pth",
                        help="Path to Stage 1 detector weights (.pth)")
    parser.add_argument("--vision-tower", type=str,
                        default="./utils/weights/vision_tower.pth",
                        help="Path to CLIP vision tower weights")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda). Auto-detected if not set.")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.weights, args.vision_tower, device)

    # Preprocess image
    print(f"[4/4] Processing image: {args.image}")
    image_tensor = preprocess_image(args.image).to(device)

    # Run inference
    with torch.no_grad():
        output = model(image_tensor, return_dict=True)
        pred = output['pred']
        probs = F.softmax(pred, dim=-1)

    real_prob = probs[0, 0].item()
    fake_prob = probs[0, 1].item()

    print("\n" + "=" * 50)
    print("       M2F2-Det DETECTION RESULT")
    print("=" * 50)
    print(f"  Image: {os.path.basename(args.image)}")
    print(f"  Prediction: {'REAL' if real_prob > fake_prob else 'FAKE'}")
    print(f"  Confidence: {max(real_prob, fake_prob) * 100:.1f}%")
    print(f"  Real prob:  {real_prob * 100:.1f}%")
    print(f"  Fake prob:  {fake_prob * 100:.1f}%")
    print("=" * 50)

    return 0 if real_prob > fake_prob else 1


if __name__ == "__main__":
    sys.exit(main())
