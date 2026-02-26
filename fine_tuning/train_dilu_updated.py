import argparse
import os
import sys
import json
import re
from datetime import datetime


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
ACTION_ID_PATTERN = re.compile(r"Response to user:\s*\#{4}\s*([0-4])", re.IGNORECASE)

def _resolve_data_file(path_str: str) -> str:
    """Resolve dataset path across local runs and this sandbox."""
    if os.path.exists(path_str):
        return path_str
    sandbox_path = os.path.join("/mnt/data", os.path.basename(path_str))
    if os.path.exists(sandbox_path):
        return sandbox_path
    return path_str


def _extract_action_id(output_text: str):
    match = ACTION_ID_PATTERN.search(output_text or "")
    if match:
        return int(match.group(1))
    return None


def _dataset_action_distribution(dataset):
    counts = {}
    for row in dataset:
        action = _extract_action_id(row.get("output", ""))
        key = "unknown" if action is None else str(action)
        counts[key] = counts.get(key, 0) + 1
    return counts


def train(args):
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset
    import torch

    data_file_cfg = args.data_file or DATA_FILE
    output_dir_cfg = args.output_dir or OUTPUT_DIR
    merged_model_dir_cfg = args.merged_model_dir or MERGED_MODEL_DIR
    model_name_cfg = args.model_name or MODEL_NAME
    max_steps_cfg = args.max_steps
    learning_rate_cfg = args.learning_rate
    max_seq_len_cfg = args.max_seq_length or MAX_SEQ_LENGTH
    val_ratio_cfg = args.val_ratio
    seed_cfg = args.seed

    # 1. Load Model & Tokenizer
    print(f"Loading model: {model_name_cfg}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_cfg,
        max_seq_length=max_seq_len_cfg,
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
    # Convert JSONL rows (instruction/input/output) into a Llama-3 chat transcript.
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
    data_file = _resolve_data_file(data_file_cfg)
    print(f"Loading dataset from {data_file}...")
    dataset = load_dataset("json", data_files=data_file, split="train")

    eval_dataset = None
    if val_ratio_cfg and 0 < val_ratio_cfg < 1:
        split = dataset.train_test_split(test_size=val_ratio_cfg, seed=seed_cfg, shuffle=True)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset

    train_dist = _dataset_action_distribution(train_dataset)
    eval_dist = _dataset_action_distribution(eval_dataset) if eval_dataset is not None else None
    print(f"Train rows: {len(train_dataset)} | Eval rows: {len(eval_dataset) if eval_dataset is not None else 0}")
    print(f"Train action distribution: {train_dist}")
    if eval_dist is not None:
        print(f"Eval action distribution: {eval_dist}")

    eval_enabled = eval_dataset is not None and len(eval_dataset) > 0

    # 6. Initialize Trainer
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=max_seq_len_cfg,
        dataset_num_proc=1,
        formatting_func=formatting_prompts_func,
        args=TrainingArguments(
            output_dir=output_dir_cfg,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=max_steps_cfg,
            warmup_steps=5,
            learning_rate=learning_rate_cfg,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=seed_cfg,
            report_to="none",
            evaluation_strategy="steps" if eval_enabled else "no",
            eval_steps=10 if eval_enabled else None,
        ),
    )

    # 7. Train
    print("Starting training...")
    trainer.train()

    os.makedirs(output_dir_cfg, exist_ok=True)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_model": model_name_cfg,
        "data_file": data_file,
        "output_dir": output_dir_cfg,
        "merged_model_dir": merged_model_dir_cfg,
        "max_seq_length": max_seq_len_cfg,
        "max_steps": max_steps_cfg,
        "learning_rate": learning_rate_cfg,
        "seed": seed_cfg,
        "val_ratio": val_ratio_cfg,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "train_action_distribution": train_dist,
        "eval_action_distribution": eval_dist,
    }
    metadata_path = os.path.join(output_dir_cfg, "run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved training metadata: {metadata_path}")

    # --- SAVE MERGED MODEL (Crucial for Ollama) ---
    print("\nTraining Complete! Saving MERGED model...")
    print(f"Target Directory: {merged_model_dir_cfg}")
    print("(This takes a minute as it combines the base model with your changes)")

    # This saves the full model (approx 5-9 GB) in safetensors format
    # Ollama can read this folder directly without extra conversion scripts!
    model.save_pretrained_merged(
        merged_model_dir_cfg,
        tokenizer,
        save_method="merged_16bit",  # Saves as standard safetensors
    )

    print(f"SUCCESS! Full model saved to: {merged_model_dir_cfg}")
    print("Next Step: Use the tracked template in fine_tuning/modelfiles/ and set the correct FROM path to your merged model folder.")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a DiLu driving copilot model and export a merged model for Ollama.")
    parser.add_argument("--data-file", default=DATA_FILE, help="Path to cleaned JSONL dataset")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Training checkpoint output directory")
    parser.add_argument("--merged-model-dir", default=MERGED_MODEL_DIR, help="Directory for merged model export")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Base model to fine-tune")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH, help="Maximum sequence length")
    parser.add_argument("--max-steps", type=int, default=60, help="Training max steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Training learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (0 disables eval)")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for split/training")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
