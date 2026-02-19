import os
import sys
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
from unsloth import is_bfloat16_supported


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
# --- CONFIGURATION ---
# --- ORGANIZED CONFIGURATION ---
# 1. Input Data Path (Now in the data folder)
DATA_FILE = "data/gold_standard_data_clean.jsonl"

# 2. Checkpoint Directory (Intermediate steps)
OUTPUT_DIR = "fine_tuning/checkpoints"

## 3. Final Adapter Save Directory (Clean location)
#FINAL_SAVE_DIR = "fine_tuning/adapters/dilu-llama3_1-8b-v1"

# 3. Where to save the FINAL MERGED MODEL (Ready for Ollama)
MERGED_MODEL_DIR = "fine_tuning/merged_models/dilu-llama3_1-8b-v1"

MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 2048

def _resolve_data_file(path_str: str) -> str:
    """Resolve dataset path across local runs and this sandbox."""
    if os.path.exists(path_str):
        return path_str
    sandbox_path = os.path.join("/mnt/data", os.path.basename(path_str))
    if os.path.exists(sandbox_path):
        return sandbox_path
    return path_str


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
    # Convert JSONL rows (instruction/input/checkpoints) into a Llama-3 chat transcript.
    # Returning {"text": [...]} is the most version-robust behavior for TRL's SFTTrainer.
    def formatting_prompts_func(examples):
        texts = []
        for instruction, input_text, output_text in zip(
                examples["instruction"], examples["input"], examples["output"]
        ):
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
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
    data_file = _resolve_data_file(DATA_FILE)
    print(f"Loading dataset from {data_file}...")
    dataset = load_dataset("json", data_files=data_file, split="train")

    # 6. Initialize Trainer
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        formatting_func=formatting_prompts_func,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=60,
            warmup_steps=5,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )

    # 7. Train
    print("Starting training...")
    trainer.train()

    # --- SAVE MERGED MODEL (Crucial for Ollama) ---
    print("\nTraining Complete! Saving MERGED model...")
    print(f"Target Directory: {MERGED_MODEL_DIR}")
    print("(This takes a minute as it combines the base model with your changes)")

    # This saves the full model (approx 5-9 GB) in safetensors format
    # Ollama can read this folder directly without extra conversion scripts!
    model.save_pretrained_merged(
        MERGED_MODEL_DIR,
        tokenizer,
        save_method="merged_16bit",  # Saves as standard safetensors
    )

    print(f"SUCCESS! Full model saved to: {MERGED_MODEL_DIR}")
    print("Next Step: Create a 'Modelfile' with 'FROM ./fine_tuning/merged_model/dilu-llama3-merged'")


if __name__ == "__main__":
    train()