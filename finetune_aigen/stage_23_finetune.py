"""
Stage 2 & 3 Fine-tuning for AI-Generated Image Detection.

Stage 2: Align deepfake_projector with new encoder features (frozen LLM)
Stage 3: LoRA fine-tune LLM for detection + explanation (unfrozen projectors)

This script adapts the original DeepSpeed-based training to work on a
single Colab T4 GPU with 4-bit quantization.

Usage:
    # Stage 2 (projector alignment)
    python finetune_aigen/stage_23_finetune.py --stage 2 \\
        --model_path CHELSEA234/llava-v1.5-7b-M2F2-Det \\
        --deepfake_ckpt ./checkpoints/stage_1_aigen/M2F2_Det_densenet121_aigen.pth \\
        --data_path ./utils/DDVQA_split/aigen/train_DDVQA_aigen_judge_only.json \\
        --image_folder ./utils/DDVQA_images/aigen/train \\
        --output_dir ./checkpoints/stage_2_aigen

    # Stage 3 (LoRA fine-tuning)
    python finetune_aigen/stage_23_finetune.py --stage 3 \\
        --model_path ./checkpoints/stage_2_aigen \\
        --deepfake_ckpt ./checkpoints/stage_1_aigen/M2F2_Det_densenet121_aigen.pth \\
        --data_path ./utils/DDVQA_split/aigen/train_DDVQA_aigen_full.json \\
        --image_folder ./utils/DDVQA_images/aigen/train \\
        --output_dir ./checkpoints/stage_3_aigen
"""

import os
import sys
import json
import copy
import logging
import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers
from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset for DDVQA-format conversations
# ============================================================================

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEEPFAKE_TOKEN_INDEX = -300

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_DEEPFAKE_TOKEN = "<deepfake>"


class DDVQADataset(Dataset):
    """Dataset for DDVQA-format conversational data."""
    
    def __init__(self, data_path: str, image_folder: str, tokenizer, 
                 image_processor, deepfake_processor, max_length: int = 2048):
        self.data = json.load(open(data_path, 'r'))
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.deepfake_processor = deepfake_processor
        self.max_length = max_length
        
        print(f"  Loaded {len(self.data)} conversation samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        image_file = entry['image']
        conversations = entry['conversations']
        
        # Load and process image
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        
        # Process for CLIP vision tower
        clip_image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        # Process for deepfake encoder (ImageNet normalization)
        deepfake_image = self.deepfake_processor(image)
        
        # Tokenize conversation
        input_ids, labels = self._tokenize_conversation(conversations)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'clip_image': clip_image,
            'deepfake_image': deepfake_image,
        }
    
    def _tokenize_conversation(self, conversations):
        """Tokenize a multi-turn conversation with appropriate masking."""
        
        # Build the full conversation text
        # Format: "USER: <content> ASSISTANT: <response>"
        input_ids = []
        labels = []
        
        for i, turn in enumerate(conversations):
            role = turn['from']
            content = turn['value']
            
            # Replace special tokens
            content = content.replace(DEFAULT_IMAGE_TOKEN, "")
            content = content.replace(DEFAULT_DEEPFAKE_TOKEN, "")
            content = content.strip()
            
            if role == 'human':
                # Add "USER: " prefix
                prefix = "USER: "
                if i == 0:
                    # First turn: add image and deepfake tokens
                    prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                    content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                    
                    turn_ids = [self.tokenizer.bos_token_id] if i == 0 else []
                    turn_ids += prefix_ids
                    turn_ids += [IMAGE_TOKEN_INDEX]  # placeholder for image features
                    turn_ids += [DEEPFAKE_TOKEN_INDEX]  # placeholder for deepfake features
                    turn_ids += content_ids
                else:
                    full_text = f" {prefix}{content}"
                    turn_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
                
                # Mask human turns (don't compute loss on user input)
                input_ids.extend(turn_ids)
                labels.extend([IGNORE_INDEX] * len(turn_ids))
            
            elif role == 'gpt':
                # Add "ASSISTANT: " prefix
                full_text = f" ASSISTANT: {content}"
                turn_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
                
                # Only compute loss on assistant responses (after "ASSISTANT: ")
                assistant_prefix = " ASSISTANT: "
                prefix_ids = self.tokenizer.encode(assistant_prefix, add_special_tokens=False)
                prefix_len = len(prefix_ids)
                
                input_ids.extend(turn_ids)
                labels.extend([IGNORE_INDEX] * prefix_len + turn_ids[prefix_len:])
        
        # Add EOS
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        
        # Truncate
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        
        # Pad
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id or 0] * padding_length
        labels = labels + [IGNORE_INDEX] * padding_length
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch):
    """Custom collate function for the dataset."""
    input_ids = torch.stack([b['input_ids'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    clip_images = torch.stack([b['clip_image'] for b in batch])
    deepfake_images = torch.stack([b['deepfake_image'] for b in batch])
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'images': clip_images,
        'deepfake_images': deepfake_images,
    }


# ============================================================================
# Model Loading with LoRA
# ============================================================================

def load_model_for_finetuning(model_path: str, deepfake_ckpt: str, 
                               stage: int, device: str = 'cuda'):
    """
    Load the M2F2-Det model for fine-tuning.
    
    Stage 2: Load full model, freeze everything except deepfake_projector
    Stage 3: Load model with LoRA, train LoRA + projectors
    """
    from llava.model.builder import load_deepfake_model
    
    kwargs = {
        "device_map": "auto",
    }
    
    # 4-bit quantization for T4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=[
            "deepfake_encoder", "deepfake_projector", 
            "lm_head", "mm_projector"
        ],
    )
    kwargs["quantization_config"] = bnb_config
    
    print(f"  Loading model from: {model_path}")
    tokenizer, model, image_processor, context_len = load_deepfake_model(
        model_path=model_path,
        model_base=None,
        model_name="llava-v1.5-7b-M2F2-Det",
        deepfake_ckpt_path=deepfake_ckpt,
        **kwargs
    )
    
    # Setup training mode based on stage
    if stage == 2:
        # Stage 2: Only train deepfake_projector
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        for name, param in model.named_parameters():
            if 'deepfake_projector' in name:
                param.requires_grad = True
                print(f"    [TRAIN] {name}: {param.shape}")
    
    elif stage == 3:
        # Stage 3: LoRA on LLM + projectors
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            os.system(f"{sys.executable} -m pip install peft")
            from peft import LoraConfig, get_peft_model, TaskType
        
        # First freeze everything
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=64,  # Reduced from 128 for T4
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        
        # Unfreeze projectors
        for name, param in model.named_parameters():
            if 'deepfake_projector' in name or 'mm_projector' in name:
                param.requires_grad = True
        
        model.print_trainable_parameters()
    
    # Create deepfake processor (ImageNet normalization for DenseNet)
    from torchvision import transforms
    deepfake_processor = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return tokenizer, model, image_processor, deepfake_processor


# ============================================================================
# Training Loop
# ============================================================================

def train_stage(model, tokenizer, train_dataset, args, stage: int):
    """Simple training loop for Stage 2 or 3."""
    
    from torch.cuda.amp import autocast, GradScaler
    
    device = next(model.parameters()).device
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    
    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * 0.03)
    
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    scaler = GradScaler()
    
    print(f"\n{'='*60}")
    print(f"Stage {stage} Training")
    print(f"  Samples: {len(train_dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Grad accumulation: {args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Total steps: {total_steps}")
    print(f"{'='*60}\n")
    
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            images = batch['images'].to(device)
            deepfake_images = batch['deepfake_images'].to(device)
            
            with autocast():
                # The model's forward handles image/deepfake token replacement
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    images=images,
                )
                loss = outputs.loss / args.grad_accum
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            total_loss += loss.item() * args.grad_accum
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        
        if stage == 3:
            # Save LoRA weights
            model.save_pretrained(save_path)
        else:
            # Save full model state
            torch.save({
                'epoch': epoch,
                'model_state_dict': {k: v for k, v in model.state_dict().items() 
                                     if 'deepfake_projector' in k},
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_path, 'checkpoint.pth'))
        
        tokenizer.save_pretrained(save_path)
        print(f"  Saved checkpoint to: {save_path}")
    
    # Save final
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    if stage == 3:
        model.save_pretrained(final_path)
    else:
        torch.save(model.state_dict(), os.path.join(final_path, 'model.pth'))
    tokenizer.save_pretrained(final_path)
    print(f"\n  Final model saved to: {final_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 2/3 Fine-tuning for AI-Gen Detection')
    parser.add_argument('--stage', type=int, required=True, choices=[2, 3],
                       help='Training stage (2=projector alignment, 3=LoRA fine-tuning)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to base model (HuggingFace ID or local path)')
    parser.add_argument('--deepfake_ckpt', type=str, required=True,
                       help='Path to fine-tuned M2F2Det checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to DDVQA-format JSON training data')
    parser.add_argument('--image_folder', type=str, required=True,
                       help='Path to image folder')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--grad_accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--max_length', type=int, default=2048, help='Max sequence length')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    
    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model for Stage {args.stage} fine-tuning...")
    print(f"{'='*60}")
    
    tokenizer, model, image_processor, deepfake_processor = load_model_for_finetuning(
        args.model_path, args.deepfake_ckpt, args.stage
    )
    
    # Load dataset
    print("\nLoading dataset...")
    train_dataset = DDVQADataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        deepfake_processor=deepfake_processor,
        max_length=args.max_length,
    )
    
    # Train
    train_stage(model, tokenizer, train_dataset, args, args.stage)
    
    print(f"\n{'='*60}")
    print(f"Stage {args.stage} fine-tuning complete!")
    print(f"Output: {args.output_dir}")
    if args.stage == 2:
        print("\nNext: Run Stage 3 fine-tuning:")
        print(f"  python finetune_aigen/stage_23_finetune.py --stage 3 \\")
        print(f"    --model_path {args.output_dir}/final \\")
        print(f"    --deepfake_ckpt {args.deepfake_ckpt} \\")
        print(f"    --data_path ./utils/DDVQA_split/aigen/train_DDVQA_aigen_full.json \\")
        print(f"    --image_folder {args.image_folder} \\")
        print(f"    --output_dir ./checkpoints/stage_3_aigen")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
