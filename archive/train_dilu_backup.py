import os
import sys

# --- WINDOWS PATH FIX (CRITICAL) ---
# We force a short temp directory to prevent "FileNotFoundError" in Triton/Inductor
# 1. Create a short temp folder
short_temp = "C:\\ui"
if not os.path.exists(short_temp):
    os.makedirs(short_temp, exist_ok=True)

# 2. Tell system to use it
os.environ["TMPDIR"] = short_temp
os.environ["TEMP"] = short_temp
os.environ["TMP"] = short_temp
os.environ["TORCHINDUCTOR_CACHE_DIR"] = short_temp
os.environ["TRITON_CACHE_DIR"] = short_temp
# -----------------------------------

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# --- CONFIGURATION ---
MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct"
DATA_FILE = "fine_tuning/gold_standard_data_clean.jsonl"
OUTPUT_DIR = "fine_tuning/checkpoints/meta-llama"
MAX_SEQ_LENGTH = 4096


def train():
    # 1. Load Model
    print(f"Loading model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. Setup Tokenizer
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )

    # 3. Formatting Function
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["checkpoints"]

        # Handle single example vs batch
        if isinstance(instructions, str):
            instructions = [instructions]
            inputs = [inputs]
            outputs = [outputs]

        texts = []
        for instruction, input_text, output_text in zip(instructions, inputs, outputs):
            conversation = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text},
            ]

            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)

        return texts

    # 4. LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        # FIX: Change "unsloth" to True for Windows stability
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    # 5. Load Data
    print(f"Loading dataset from {DATA_FILE}...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train", keep_in_memory=True)

    # 6. Initialize Trainer
    print("Starting training setup...")
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
            max_steps=60,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",
            report_to="none",
        ),
    )

    # 7. Train
    print("Training started...")
    trainer.train()

    # 8. Export to GGUF
    print("Exporting to GGUF format...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # This merges the LoRA adapters into the base model and saves as standard Safetensors
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
    print(f"Done! Merged model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()