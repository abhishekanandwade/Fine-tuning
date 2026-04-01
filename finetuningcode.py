# import torch
# from datasets import load_dataset
# from unsloth import FastLanguageModel
# from transformers import TrainingArguments
# from trl import SFTTrainer

# # ==========================
# # Configuration
# # ==========================

# MODEL_NAME = "Snowflake/Arctic-Text2SQL-R1-7B"
# DATASET_PATH = "data/combined_lora_dataset.jsonl"  # your dataset file
# OUTPUT_DIR = "arctic_text2sql_lora"

# MAX_SEQ_LENGTH = 4096

# # ==========================
# # Load model with Unsloth
# # ==========================

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = MODEL_NAME,
#     max_seq_length = MAX_SEQ_LENGTH,
#     dtype = None,
#     load_in_4bit = True,
# )

# # ==========================
# # Add LoRA adapters
# # ==========================

# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 16,
#     target_modules = [
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#         "gate_proj",
#         "up_proj",
#         "down_proj",
#     ],
#     lora_alpha = 32,
#     lora_dropout = 0,
#     bias = "none",
#     use_gradient_checkpointing = "unsloth",
# )

# # ==========================
# # Load dataset
# # ==========================

# dataset = load_dataset(
#     "json",
#     data_files = DATASET_PATH,
#     split = "train",
# )

# # ==========================
# # Convert your format to prompt
# # ==========================

# def format_dataset(example):

#     system_prompt = example["system"]

#     user_prompt = ""
#     assistant_response = ""

#     for msg in example["conversations"]:
#         if msg["role"] == "user":
#             user_prompt = msg["content"]
#         elif msg["role"] == "assistant":
#             assistant_response = msg["content"]

#     text = (
#         f"<|system|>\n{system_prompt}\n"
#         f"<|user|>\n{user_prompt}\n"
#         f"<|assistant|>\n{assistant_response}{tokenizer.eos_token}"
#     )

#     return {"text": text}


# dataset = dataset.map(format_dataset)

# # ==========================
# # Tokenization
# # ==========================

# def tokenize_function(example):
#     return tokenizer(
#         example["text"],
#         truncation = True,
#         padding = "max_length",
#         max_length = MAX_SEQ_LENGTH,
#     )

# dataset = dataset.map(tokenize_function, batched=True)

# # ==========================
# # Trainer
# # ==========================

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     max_seq_length=MAX_SEQ_LENGTH,
#     packing=True,

#     args=TrainingArguments(
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=8,
#         warmup_steps=50,
#         max_steps=2000,
#         learning_rate=2e-4,
#         logging_steps=10,
#         optim="adamw_8bit",
#         weight_decay=0.01,
#         lr_scheduler_type="linear",
#         bf16=True,         
#         fp16=False,         
#         output_dir=OUTPUT_DIR,
#         save_steps=200,
#         save_total_limit=3,
#     ),
# )

# # ==========================
# # Train
# # ==========================

# trainer.train()

# # ==========================
# # Save LoRA adapters
# # ==========================

# model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)

# print("Training complete. LoRA adapter saved to:", OUTPUT_DIR)

import logging
import os

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, set_seed
from trl import SFTTrainer
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

# ==========================
# Logging
# ==========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==========================
# Configuration
# ==========================

MODEL_NAME = "Snowflake/Arctic-Text2SQL-R1-7B"
DATASET_PATH = "data/combined_lora_dataset.jsonl"
OUTPUT_DIR = "arctic_text2sql_lora"

MAX_SEQ_LENGTH = 4096
SEED = 42

# LoRA hyper-parameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj", "v_proj",
]

# Training hyper-parameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_STEPS = 50
MAX_STEPS = 2000
LEARNING_RATE = 2e-4
LOGGING_STEPS = 10
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
NEFTUNE_NOISE_ALPHA = 5  # NEFTune regularisation for better generalisation
EVAL_SPLIT_RATIO = 0.05  # 5 % held-out for validation

# ChatML response template for masking 
# Loss is only computed on the assistant's response, not on system/user prompts.
RESPONSE_TEMPLATE = "<|im_start|>assistant"
INSTRUCTION_TEMPLATE = "<|im_start|>user"


def load_model_and_tokenizer():
    """Load the base model with 4-bit quantisation via Unsloth."""
    logger.info("Loading model: %s", MODEL_NAME)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        attn_implementation="sdpa",
    )
    return model, tokenizer


def apply_lora(model):
    """Attach LoRA adapters to the model."""
    logger.info("Applying LoRA adapters (r=%d, alpha=%d)", LORA_R, LORA_ALPHA)
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  
    )
    return model


def prepare_dataset(tokenizer):
    """Load the JSONL dataset, format prompts, and create train/eval splits."""
    logger.info("Loading dataset from: %s", DATASET_PATH)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    dataset = load_dataset(
        "json",
        data_files=DATASET_PATH,
        split="train",
    )

    def format_example(example):
        """Convert a single example into the ChatML training prompt format."""
        system_prompt = example.get("system", "")
        user_prompt = ""
        assistant_response = ""

        for msg in example.get("conversations", []):
            if msg["role"] == "user":
                user_prompt = msg["content"]
            elif msg["role"] == "assistant":
                assistant_response = msg["content"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token

        return {"text": text}

    dataset = dataset.map(format_example, desc="Formatting prompts")

    # Train / eval split
    split = dataset.train_test_split(test_size=EVAL_SPLIT_RATIO, seed=SEED)
    logger.info(
        "Dataset ready — %d train samples, %d eval samples",
        len(split["train"]),
        len(split["test"]),
    )
    return split["train"], split["test"]


def _build_response_masking_collator(tokenizer):
    """Build a data collator that masks prompt tokens (system + user) from the loss.

    Only the assistant response tokens contribute to the training loss.
    Uses token-ID matching for robustness against tokenisation edge-cases.
    """
    response_ids = tokenizer.encode(
        RESPONSE_TEMPLATE, add_special_tokens=False
    )
    instruction_ids = tokenizer.encode(
        INSTRUCTION_TEMPLATE, add_special_tokens=False
    )
    logger.info(
        "Response masking — response_template_ids=%s, instruction_template_ids=%s",
        response_ids,
        instruction_ids,
    )
    return DataCollatorForCompletionOnlyLM(
        response_template=response_ids,
        instruction_template=instruction_ids,
        tokenizer=tokenizer,
        mlm=False,
    )


def build_trainer(model, tokenizer, train_dataset, eval_dataset):
    """Configure and return the SFTTrainer with response masking."""
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type="linear",
        bf16=use_bf16,
        fp16=not use_bf16,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        seed=SEED,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        neftune_noise_alpha=NEFTUNE_NOISE_ALPHA,
        report_to="none", 
    )

    collator = _build_response_masking_collator(tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,  # packing must be False when using completion-only masking
        args=training_args,
    )
    return trainer


def main():
    set_seed(SEED)

    # 1. Model & tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 2. LoRA
    model = apply_lora(model)
    FastLanguageModel.for_training(model)

    # 3. Dataset
    train_dataset, eval_dataset = prepare_dataset(tokenizer)

    # 4. Trainer
    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset)

    # 5. Train
    logger.info("Starting training …")
    trainer.train()

    # 6. Save LoRA adapters
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Training complete. LoRA adapter saved to: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()