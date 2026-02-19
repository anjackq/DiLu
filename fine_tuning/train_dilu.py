import os
import sys
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# --- WINDOWS PATH FIX (CRITICAL) ---
# We force a short temp directory to prevent "FileNotFoundError" in Triton/Inductor
# 1. Create a short temp folder
short_temp = "C:\\ui"
if not os.path.exists(short_temp):
    try:
        os.makedirs(short_temp, exist_ok=True)
    except:
        # If C:\ui fails (permissions), try user home but short
        short_temp = os.path.join(os.environ.get("HOMEDRIVE", "C:"), "\\temp_ui")
        os.makedirs(short_temp, exist_ok=True)

# 2. Tell system to use it for EVERYTHING
os.environ["TMPDIR"] = short_temp
os.environ["TEMP"] = short_temp
os.environ["TMP"] = short_temp
os.environ["TORCHINDUCTOR_CACHE_DIR"] = short_temp
os.environ["TRITON_CACHE_DIR"] = short_temp
# -----------------------------------

# --- CONFIGURATION ---
# --- ORGANIZED CONFIGURATION ---
# 1. Input Data Path (Now in the data folder)
DATA_FILE = "data/gold_standard_data_clean.jsonl"

# 2. Checkpoint Directory (Intermediate steps)
OUTPUT_DIR = "fine_tuning/checkpoints"

# 3. Final Adapter Save Directory (Clean location)
FINAL_SAVE_DIR = "fine_tuning/adapters/dilu-llama3_1-8b-v1"

MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 2048


def train():
    # 1. Load Model & Tokenizer
    print(f"Loading model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. Setup Tokenizer for Llama 3
    # This automatically maps your data to the specific Llama-3 prompt format
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )

    # 3. Define Formatting Function
    # This converts your JSONL rows into the specific text structure the model trains on
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output_text in zip(instructions, inputs, outputs):
            # Must match the "Alpaca" style prompts we generated earlier
            text = f"""<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"""
            texts.append(text)
        return texts

    # 4. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Use "unsloth" for memory optimization
        random_state=3407,
    )

    # 5. Load Data
    print(f"Loading dataset from {DATA_FILE}...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # 6. Initialize Trainer
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        formatting_func=formatting_prompts_func,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,  # 60 steps is a quick run; increase to ~300 for better results
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",
            report_to="none",
            seed=3407,
        ),
    )

    # 7. Train
    print("Starting training...")
    trainer.train()

    # 6. Save Adapters ONLY (Safe for Windows)
    # This skips the problematic GGUF conversion loop
    print(f"Saving final LoRA adapters to '{FINAL_SAVE_DIR}'...")
    model.save_pretrained(FINAL_SAVE_DIR)
    tokenizer.save_pretrained(FINAL_SAVE_DIR)
    print("SUCCESS! Adapters saved.")


if __name__ == "__main__":
    train()