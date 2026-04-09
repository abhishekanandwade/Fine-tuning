"""
fine_tune_go_reviewer.py — QLoRA Fine-tuning Script for Go Code Reviewer

Fine-tunes a code LLM (deepseek-coder / qwen2.5-coder) on Go code review
examples using QLoRA (4-bit quantization + LoRA adapters).

Usage:
    python training/fine_tune_go_reviewer.py
    python training/fine_tune_go_reviewer.py --config training/training_config.yaml
    python training/fine_tune_go_reviewer.py --model deepseek-ai/deepseek-coder-7b-instruct-v1.5
"""

import argparse
import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ── Default Configuration ────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model": {
        "name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "trust_remote_code": True,
    },
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    },
    "lora": {
        "r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    "training": {
        "output_dir": "./go-reviewer-model",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 0.0002,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "max_seq_length": 8192,
        "packing": False,
        "fp16": False,
        "bf16": True,
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "report_to": "wandb",
    },
    "dataset": {
        "train_file": "dataset/processed/train.jsonl",
        "validation_file": "dataset/processed/validation.jsonl",
    },
    "output": {
        "final_model_dir": "./go-reviewer-final",
    },
}


def load_config(config_path: str = None) -> dict:
    """Load training config from YAML file or use defaults."""
    config = DEFAULT_CONFIG.copy()

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)

        # Deep merge user config into defaults
        for section, values in user_config.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(values)
            else:
                config[section] = values

        print(f"[INFO] Loaded config from {config_path}")
    else:
        print("[INFO] Using default configuration")

    return config


def get_compute_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def setup_quantization(config: dict) -> BitsAndBytesConfig:
    """Create BitsAndBytesConfig from config."""
    quant_cfg = config["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=get_compute_dtype(quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )


def setup_model_and_tokenizer(config: dict, bnb_config: BitsAndBytesConfig):
    """Load and prepare model + tokenizer."""
    model_cfg = config["model"]
    model_name = model_cfg["name"]

    print(f"[INFO] Loading model: {model_name}")
    print(f"[INFO] Quantization: 4-bit NF4 with double quantization")

    # Use Flash Attention 2 if available, otherwise fall back to sdpa/eager
    attn_impl = None
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            print("[INFO] Using Flash Attention 2")
        except ImportError:
            attn_impl = "sdpa"
            print("[INFO] Flash Attention not installed, falling back to SDPA")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        attn_implementation=attn_impl,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def setup_lora(config: dict, model):
    """Apply LoRA adapters to the model."""
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"[INFO] Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model


def load_datasets(config: dict):
    """Load training and validation datasets."""
    ds_cfg = config["dataset"]

    data_files = {"train": ds_cfg["train_file"]}
    if os.path.exists(ds_cfg.get("validation_file", "")):
        data_files["validation"] = ds_cfg["validation_file"]

    dataset = load_dataset("json", data_files=data_files)

    print(f"[INFO] Training examples: {len(dataset['train'])}")
    if "validation" in dataset:
        print(f"[INFO] Validation examples: {len(dataset['validation'])}")

    return dataset


def format_chat_template(example, tokenizer):
    """
    Format a training example using the tokenizer's chat template.
    Falls back to a simple concat if no chat template is available.
    """
    messages = example.get("messages", [])

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}
        except Exception:
            pass

    # Fallback: simple formatting
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"### System:\n{content}\n")
        elif role == "user":
            parts.append(f"### User:\n{content}\n")
        elif role == "assistant":
            parts.append(f"### Assistant:\n{content}\n")
    return {"text": "\n".join(parts)}


def train(config: dict):
    """Main training function."""
    train_cfg = config["training"]

    # 1. Setup quantization
    bnb_config = setup_quantization(config)

    # 2. Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config, bnb_config)

    # 3. Apply LoRA
    model = setup_lora(config, model)

    # 4. Load datasets
    dataset = load_datasets(config)

    # 5. Format dataset with chat template
    print("[INFO] Formatting dataset with chat template...")
    formatted_train = dataset["train"].map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    formatted_val = None
    if "validation" in dataset:
        formatted_val = dataset["validation"].map(
            lambda x: format_chat_template(x, tokenizer),
            remove_columns=dataset["validation"].column_names,
        )

    # 6. Training configuration
    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        packing=train_cfg["packing"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        report_to=train_cfg.get("report_to", "none"),
        dataset_text_field="text",
    )

    # 7. Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=formatted_train,
        eval_dataset=formatted_val,
        max_seq_length=train_cfg["max_seq_length"],
    )

    # 8. Train
    print("\n" + "=" * 60)
    print("[INFO] Starting training...")
    print("=" * 60 + "\n")

    trainer.train()

    # 9. Save final model
    final_dir = config["output"]["final_model_dir"]
    print(f"\n[INFO] Saving final model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\n[SUCCESS] Training complete! Model saved to {final_dir}")
    print(f"[INFO] To use: load with AutoModelForCausalLM.from_pretrained('{final_dir}')")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a code LLM for Go code review using QLoRA"
    )
    parser.add_argument(
        "--config",
        default="training/training_config.yaml",
        help="Path to training config YAML (default: training/training_config.yaml)",
    )
    parser.add_argument(
        "--model",
        help="Override model name (e.g., deepseek-ai/deepseek-coder-33b-instruct)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override per-device batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.model:
        config["model"]["name"] = args.model
    if args.epochs:
        config["training"]["num_train_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["per_device_train_batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.no_wandb:
        config["training"]["report_to"] = "none"

    # Print config summary
    print("\n" + "=" * 60)
    print("Go Code Review Model — Fine-tuning with QLoRA")
    print("=" * 60)
    print(f"  Model:        {config['model']['name']}")
    print(f"  LoRA rank:    {config['lora']['r']}")
    print(f"  Epochs:       {config['training']['num_train_epochs']}")
    print(f"  Batch size:   {config['training']['per_device_train_batch_size']}")
    print(f"  LR:           {config['training']['learning_rate']}")
    print(f"  Max seq len:  {config['training']['max_seq_length']}")
    print(f"  Output:       {config['output']['final_model_dir']}")
    print("=" * 60 + "\n")

    train(config)


if __name__ == "__main__":
    main()
