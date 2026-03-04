"""
Evaluation Pipeline for AI-Generated Image Detection.

Evaluates both Stage 1 (binary classifier) and Stage 3 (LLM-based detection)
on a held-out test set of real and AI-generated images.

Usage:
    # Evaluate Stage 1 model
    python finetune_aigen/evaluate.py --mode stage1 \\
        --model_ckpt ./checkpoints/stage_1_aigen/best_model.pth \\
        --data_dir ./finetune_aigen/data/test

    # Evaluate Stage 3 (LLM) model  
    python finetune_aigen/evaluate.py --mode stage3 \\
        --model_path CHELSEA234/llava-v1.5-7b-M2F2-Det \\
        --deepfake_ckpt ./checkpoints/stage_1_aigen/M2F2_Det_densenet121_aigen.pth \\
        --data_dir ./finetune_aigen/data/test

    # Evaluate on a single image
    python finetune_aigen/evaluate.py --mode single \\
        --model_ckpt ./checkpoints/stage_1_aigen/best_model.pth \\
        --image_path /path/to/image.jpg
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve
)


# ============================================================================
# Stage 1 Evaluation
# ============================================================================

def evaluate_stage1(model_ckpt: str, data_dir: str, batch_size: int = 16, 
                    device: str = 'cuda', output_dir: str = None):
    """Evaluate the Stage 1 M2F2Det binary classifier."""
    
    from finetune_aigen.stage_1_finetune import (
        AIGenImageDataset, get_val_transforms, load_m2f2det_model
    )
    
    print("\n" + "="*60)
    print("Stage 1 Evaluation: M2F2Det Binary Classifier")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    model = load_m2f2det_model(model_ckpt, device=device)
    model.eval()
    
    # Load dataset
    print("\n2. Loading test dataset...")
    test_dataset = AIGenImageDataset(data_dir, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"   {len(test_dataset)} test samples")
    
    # Run inference
    print("\n3. Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            with torch.cuda.amp.autocast():
                output = model(images)
                if isinstance(output, dict):
                    logits = output['pred']
                elif isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
            
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())
    
    # Compute metrics
    results = compute_metrics(all_labels, all_preds, all_probs)
    print_results(results, "Stage 1")
    
    if output_dir:
        save_results(results, all_labels, all_preds, all_probs, output_dir, "stage1")
    
    return results


# ============================================================================
# Stage 3 (LLM) Evaluation (simplified for Colab)
# ============================================================================

def evaluate_stage3_simple(model_path: str, deepfake_ckpt: str, data_dir: str,
                          device: str = 'cuda', output_dir: str = None):
    """
    Evaluate Stage 3 LLM model on detection task.
    Uses the same approach as the inference notebook.
    """
    print("\n" + "="*60)
    print("Stage 3 Evaluation: LLM-based Detection")
    print("="*60)
    
    # This requires the full LLM loading (from notebook approach)
    # For simplicity, we'll just use the binary output from generate()
    
    from llava.model.builder import load_deepfake_model
    from llava.mm_utils import tokenizer_image_token
    from transformers import BitsAndBytesConfig
    
    # Load model (4-bit quantized)
    print("\n1. Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["deepfake_encoder", "deepfake_projector", "lm_head", "mm_projector"],
    )
    
    tokenizer, model, image_processor, context_len = load_deepfake_model(
        model_path=model_path,
        model_base=None,
        model_name="llava-v1.5-7b-M2F2-Det",
        deepfake_ckpt_path=deepfake_ckpt,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model.eval()
    
    # Setup deepfake processor (ImageNet normalization)
    deepfake_processor = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Detection prompt
    prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n <deepfake>\n Determine the authenticity. Is the image real or fake? ###Assistant:"
    
    # Collect test images
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    test_images = []
    for label_name, label_id in [('real', 0), ('aigenerated', 1)]:
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        for f in sorted(os.listdir(label_dir)):
            if os.path.splitext(f)[1].lower() in img_exts:
                test_images.append((os.path.join(label_dir, f), label_id))
    
    print(f"\n2. Running detection on {len(test_images)} images...")
    
    all_labels = []
    all_preds = []
    results_detail = []
    
    for img_path, label in tqdm(test_images, desc="Detecting"):
        image = Image.open(img_path).convert('RGB')
        
        # Process images
        clip_image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        df_image = deepfake_processor(image)
        
        # Tokenize prompt
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX=-200, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=clip_image.unsqueeze(0).half().to(model.device),
                deepfake_images=df_image.unsqueeze(0).half().to(model.device),
                do_sample=False,
                max_new_tokens=50,
            )
        
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
        
        # Parse prediction
        pred = 1 if 'fake' in response else 0
        
        all_labels.append(label)
        all_preds.append(pred)
        results_detail.append({
            'image': img_path,
            'label': 'fake' if label == 1 else 'real',
            'prediction': 'fake' if pred == 1 else 'real',
            'response': response,
            'correct': pred == label
        })
    
    # Compute metrics (no probabilities for LLM, just binary)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='binary') * 100
    precision = precision_score(all_labels, all_preds, average='binary') * 100
    recall = recall_score(all_labels, all_preds, average='binary') * 100
    
    results = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'total_samples': len(all_labels),
        'correct': sum(1 for l, p in zip(all_labels, all_preds) if l == p),
    }
    
    print(f"\n{'='*60}")
    print(f"Stage 3 (LLM) Results:")
    print(f"  Accuracy:  {accuracy:.1f}%")
    print(f"  F1 Score:  {f1:.1f}%")
    print(f"  Precision: {precision:.1f}%")
    print(f"  Recall:    {recall:.1f}%")
    print(f"{'='*60}")
    
    # Show confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n  Confusion Matrix:")
    print(f"                Pred Real  Pred Fake")
    print(f"  Actual Real:  {cm[0][0]:>9}  {cm[0][1]:>9}")
    print(f"  Actual Fake:  {cm[1][0]:>9}  {cm[1][1]:>9}")
    
    # Show some failures
    failures = [r for r in results_detail if not r['correct']]
    if failures:
        print(f"\n  Failed predictions ({len(failures)}/{len(results_detail)}):")
        for r in failures[:10]:
            print(f"    {Path(r['image']).name}: actual={r['label']}, pred={r['prediction']}")
            print(f"      Response: {r['response'][:100]}...")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'stage3_results.json'), 'w') as f:
            json.dump({'metrics': results, 'details': results_detail}, f, indent=2)
    
    return results


# ============================================================================
# Single Image Evaluation
# ============================================================================

def evaluate_single_image(model_ckpt: str, image_path: str, device: str = 'cuda'):
    """Run Stage 1 detection on a single image."""
    
    from finetune_aigen.stage_1_finetune import load_m2f2det_model, get_val_transforms
    
    print(f"\nEvaluating: {image_path}")
    
    model = load_m2f2det_model(model_ckpt, device=device)
    model.eval()
    
    transform = get_val_transforms()
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model(img_tensor)
            if isinstance(output, dict):
                logits = output['pred']
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
    
    probs = F.softmax(logits, dim=-1)[0]
    pred = logits.argmax(dim=-1).item()
    
    label_map = {0: 'REAL', 1: 'AI-GENERATED (FAKE)'}
    
    print(f"\n  Prediction: {label_map[pred]}")
    print(f"  Confidence: {probs[pred].item() * 100:.1f}%")
    print(f"  P(real):    {probs[0].item() * 100:.1f}%")
    print(f"  P(fake):    {probs[1].item() * 100:.1f}%")
    
    return pred, probs.cpu().tolist()


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(labels, preds, probs=None):
    """Compute comprehensive evaluation metrics."""
    results = {
        'accuracy': accuracy_score(labels, preds) * 100,
        'f1': f1_score(labels, preds, average='binary') * 100,
        'precision': precision_score(labels, preds, average='binary', zero_division=0) * 100,
        'recall': recall_score(labels, preds, average='binary', zero_division=0) * 100,
        'total_samples': len(labels),
        'correct': sum(1 for l, p in zip(labels, preds) if l == p),
    }
    
    if probs:
        try:
            results['auc_roc'] = roc_auc_score(labels, probs) * 100
        except ValueError:
            results['auc_roc'] = 0.0
        try:
            results['avg_precision'] = average_precision_score(labels, probs) * 100
        except ValueError:
            results['avg_precision'] = 0.0
    
    # Per-class accuracy
    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        results['real_accuracy'] = cm[0][0] / max(cm[0].sum(), 1) * 100
        results['fake_accuracy'] = cm[1][1] / max(cm[1].sum(), 1) * 100
        results['confusion_matrix'] = cm.tolist()
    
    return results


def print_results(results, stage_name=""):
    """Print formatted evaluation results."""
    print(f"\n{'='*60}")
    print(f"{stage_name} Evaluation Results")
    print(f"{'='*60}")
    print(f"  Total Samples: {results['total_samples']}")
    print(f"  Correct:       {results['correct']}")
    print(f"  Accuracy:      {results['accuracy']:.1f}%")
    if 'auc_roc' in results:
        print(f"  AUC-ROC:       {results['auc_roc']:.1f}%")
    if 'avg_precision' in results:
        print(f"  Avg Precision: {results['avg_precision']:.1f}%")
    print(f"  F1 Score:      {results['f1']:.1f}%")
    print(f"  Precision:     {results['precision']:.1f}%")
    print(f"  Recall:        {results['recall']:.1f}%")
    if 'real_accuracy' in results:
        print(f"\n  Per-class:")
        print(f"    Real Accuracy: {results['real_accuracy']:.1f}%")
        print(f"    Fake Accuracy: {results['fake_accuracy']:.1f}%")
    if 'confusion_matrix' in results:
        cm = results['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"                Pred Real  Pred Fake")
        print(f"  Actual Real:  {cm[0][0]:>9}  {cm[0][1]:>9}")
        print(f"  Actual Fake:  {cm[1][0]:>9}  {cm[1][1]:>9}")
    print(f"{'='*60}")


def save_results(results, labels, preds, probs, output_dir, prefix="eval"):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, f'{prefix}_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed predictions
    details = [
        {'label': l, 'prediction': p, 'prob_fake': pr, 'correct': l == p}
        for l, p, pr in zip(labels, preds, probs)
    ]
    with open(os.path.join(output_dir, f'{prefix}_predictions.json'), 'w') as f:
        json.dump(details, f, indent=2)
    
    print(f"\n  Results saved to: {output_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate AI-Generated Image Detection')
    parser.add_argument('--mode', type=str, choices=['stage1', 'stage3', 'single'], required=True,
                       help='Evaluation mode')
    parser.add_argument('--model_ckpt', type=str, default=None,
                       help='Stage 1 model checkpoint path')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Stage 3 model path (HuggingFace ID or local)')
    parser.add_argument('--deepfake_ckpt', type=str, default=None,
                       help='Deepfake encoder checkpoint (for Stage 3)')
    parser.add_argument('--data_dir', type=str, default='./finetune_aigen/data/test',
                       help='Test data directory')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Single image path (for single mode)')
    parser.add_argument('--output_dir', type=str, default='./finetune_aigen/eval_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.mode == 'stage1':
        if not args.model_ckpt:
            parser.error("--model_ckpt is required for stage1 evaluation")
        evaluate_stage1(args.model_ckpt, args.data_dir, args.batch_size, device, args.output_dir)
    
    elif args.mode == 'stage3':
        if not args.model_path:
            parser.error("--model_path is required for stage3 evaluation")
        evaluate_stage3_simple(args.model_path, args.deepfake_ckpt, args.data_dir, device, args.output_dir)
    
    elif args.mode == 'single':
        if not args.image_path:
            parser.error("--image_path is required for single evaluation")
        if not args.model_ckpt:
            parser.error("--model_ckpt is required for single evaluation")
        evaluate_single_image(args.model_ckpt, args.image_path, device)


if __name__ == '__main__':
    main()
