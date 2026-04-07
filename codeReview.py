import json
import subprocess
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM


# -------------------------------
# 1. Load dataset
# -------------------------------
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    print(data[1])
    return data


# -------------------------------
# 2. Format dataset
# -------------------------------
def format_dataset(data, tokenizer):
    def format_prompt(example):
        messages = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": json.dumps(example["output"])},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    formatted_data = [format_prompt(item) for item in data]
    return Dataset.from_dict({"text": formatted_data})


# -------------------------------
# 3. Load model
# -------------------------------
def load_model():
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_seq_length = 2048

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False  # disable KV-cache during training

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_seq_length

    lora_config = LoraConfig(
        r=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # required for gradient checkpointing with PEFT

    return model, tokenizer, max_seq_length


# -------------------------------
# 4. Train
# -------------------------------
def train(model, tokenizer, dataset, max_seq_length):
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=max_seq_length,
            dataset_num_proc=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=25,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
        ),
    )

    return trainer.train()


# -------------------------------
# 5. Inference test
# -------------------------------
def test_model(model, tokenizer):
    model.config.use_cache = True  # re-enable KV-cache for inference
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = model.to(dtype)
    model.eval()

    messages = [
        {"role": "user", "content": "Extract product info:\n<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span></div>"}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )

    print(tokenizer.batch_decode(outputs)[0])


# -------------------------------
# 6. Merge adapter and export to GGUF
# -------------------------------
def export_to_gguf(adapter_dir="fine_tuned_model", output_gguf="/dev/shm/fine_tuned_model_q4.gguf"):
    import shutil
    merged_dir = "/dev/shm/merged_model"

    print("Merging LoRA adapter into base model...")
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        dtype=torch.float16,
        device_map="auto",
    )
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(merged_dir)

    # Fix tokenizer_config.json extra_special_tokens type mismatch
    import json as _json
    tc_path = f"{merged_dir}/tokenizer_config.json"
    with open(tc_path) as f:
        tc = _json.load(f)
    if isinstance(tc.get("extra_special_tokens"), list):
        tc["extra_special_tokens"] = {}
        with open(tc_path, "w") as f:
            _json.dump(tc, f, indent=2)

    print(f"Merged model saved to: {merged_dir}")

    print("Converting directly to Q4_0 GGUF (~4.5GB)...")
    subprocess.run(
        [
            "python", "/workspace/llama.cpp/convert_hf_to_gguf.py", merged_dir,
            "--outfile", output_gguf,
            "--outtype", "q4_0",
        ],
        check=True,
    )

    # Free ~15GB of RAM after GGUF is written
    print("Removing merged safetensors to free space...")
    shutil.rmtree(merged_dir)

    print(f"Quantized GGUF saved to: {output_gguf}")


# -------------------------------
# 7. Main entry
# -------------------------------
def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data = load_data("json_extraction_dataset_500.json")

    model, tokenizer, max_seq_length = load_model()

    dataset = format_dataset(data, tokenizer)

    train(model, tokenizer, dataset, max_seq_length)

    test_model(model, tokenizer)

    model.save_pretrained("fine_tuned_model")
    tokenizer.save_pretrained("fine_tuned_model")

    export_to_gguf("fine_tuned_model", "/dev/shm/fine_tuned_model_q4.gguf")


if __name__ == "__main__":
    main()