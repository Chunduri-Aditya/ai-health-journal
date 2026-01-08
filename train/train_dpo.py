#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) training script using LoRA.

Trains a LoRA adapter on preference pairs to improve groundedness
and reduce hallucinations in journal analysis.
"""

import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import argparse

def load_dpo_pairs(pairs_path: str) -> list:
    """Load DPO preference pairs from JSONL file."""
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs

def format_dataset(pairs: list) -> Dataset:
    """Convert pairs into HuggingFace Dataset format."""
    prompts = []
    chosen = []
    rejected = []
    
    for pair in pairs:
        prompts.append(pair['prompt'])
        chosen.append(pair['chosen'])
        rejected.append(pair['rejected'])
    
    return Dataset.from_dict({
        'prompt': prompts,
        'chosen': chosen,
        'rejected': rejected
    })

def main():
    parser = argparse.ArgumentParser(description='Train DPO model with LoRA')
    parser.add_argument('--dataset', type=str, default='train/dpo_pairs.jsonl', help='Path to DPO pairs JSONL')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Base model to fine-tune')
    parser.add_argument('--output_dir', type=str, default='train/dpo_lora_adapter', help='Output directory for LoRA adapter')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization (requires bitsandbytes)')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum training steps (overrides num_epochs if set)')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--max_prompt_len', type=int, default=1024, help='Maximum prompt length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DPO Training with LoRA")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Base model: {args.base_model}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}, Batch size: {args.batch_size}")
    print()
    
    # Load dataset
    print("Loading DPO pairs...")
    pairs = load_dpo_pairs(args.dataset)
    print(f"Loaded {len(pairs)} preference pairs")
    
    if len(pairs) == 0:
        print("❌ No pairs found. Cannot train.")
        return
    
    # Smoke test mode (if env var set)
    if os.environ.get('RUN_TRAINING_SMOKE') == '1':
        print("🧪 SMOKE TEST MODE: Limiting to 5 steps")
        if args.max_steps is None:
            args.max_steps = 5
    
    dataset = format_dataset(pairs)
    print(f"Dataset formatted: {len(dataset)} examples")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading base model from {args.base_model}...")
    model_kwargs = {}
    if args.use_4bit:
        print("Using 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model_kwargs['quantization_config'] = bnb_config
        model_kwargs['device_map'] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.use_4bit else torch.float32,
        **model_kwargs
    )
    
    # Prepare model for LoRA
    print("Preparing model for LoRA training...")
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Common for Qwen/Llama
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs if args.max_steps is None else 1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=(device == 'cuda' and not args.use_4bit),
        bf16=(device == 'cuda' and torch.cuda.is_bf16_supported() and not args.use_4bit),
        logging_steps=10,
        save_steps=100 if args.max_steps is None or args.max_steps > 100 else args.max_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        optim="paged_adamw_32bit" if args.use_4bit else "adamw_torch",
        seed=args.seed,
    )
    
    # DPO Trainer
    print("\nInitializing DPO Trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=0.1,  # DPO beta parameter
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        max_prompt_length=args.max_prompt_len,
    )
    
    # Train
    print("\nStarting training...")
    dpo_trainer.train()
    
    # Save
    print(f"\nSaving LoRA adapter to {args.output_dir}...")
    dpo_trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n✅ Training complete!")
    print(f"LoRA adapter saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
