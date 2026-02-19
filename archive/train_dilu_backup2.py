from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- WINDOWS PATH FIX (CRITICAL) ---
# We force a short temp directory to prevent "FileNotFoundError" in Triton/Inductor
# 1. Create a short temp folder
import os, sys
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

# ==========================================
#        USER CONFIGURATION VARIABLES
# ==========================================
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"  # Base model to fine-tune
DATA_FILE = "fine_tuning/gold_standard_data_clean.jsonl" # Path to your clean data
OUTPUT_DIR = "fine_tunning/checkpoints/meta-llama"                             # Folder for training checkpoints
MAX_SEQ_LENGTH = 2048                              # Context window size
# ==========================================

# --- SYSTEM CONFIGURATION ---
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage.

# 1. Load Base Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. Add LoRA Adapters (Efficient Fine-Tuning)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 3. Define Formatting Function (Alpaca Style)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["checkpoints"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise generation goes on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# 4. Load Dataset
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. Training Setup
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Adjust based on data size (60 is good for ~50-100 samples)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
    ),
)

print(f"Starting Training...")
trainer_stats = trainer.train()

# --- WINDOWS SAFE SAVING ---
print("Saving merged model (Windows Safe Mode)...")

# 1. Save the adapters only (Small file, backup)
model.save_pretrained("dilu-llama3-adapters")

# 2. Save the FULL MERGED model (Big folder, ready for conversion)
# This merges the LoRA changes into the main model so it's a standard Llama 3 model
model.save_pretrained_merged("dilu-llama3-merged", tokenizer, save_method = "merged_16bit")

print("----------------------------------------------------------------")
print("SUCCESS! The model is saved in the 'dilu-llama3-merged' folder.")
print("To use this in Ollama, you must now convert this folder to GGUF.")
print("Run the following commands in your terminal:")
print("1. git clone https://github.com/ggerganov/llama.cpp")
print("2. python llama.cpp/convert_hf_to_gguf.py dilu-llama3-merged --outfile dilu-llama3.gguf --outtype f16")
print("----------------------------------------------------------------")