# Fine-Tuning Guide

This folder contains the fine-tuning pipeline for DiLu-Ollama models.

## Pipeline Overview

1. Collect expert-labeled driving data.
2. Convert to strict training format.
3. Validate dataset schema and action format.
4. Fine-tune and export merged model weights.
5. (Recommended for local serving) Convert merged model to GGUF and create Ollama model from GGUF.

## Scripts

- `collect_data.py`: collect rule-based expert trajectories.
- `convert_data.py`: convert raw JSONL to strict `Reasoning + Response to user:#### <id>` format.
- `validate_finetune_dataset.py`: validate row schema and output format.
- `train_dilu_ollama.py`: main training/export script.
- `run_pipeline.py`: orchestrates full pipeline.
- `build_gguf.py`: convert merged HF export to GGUF (+ optional quantization and Ollama create).
- `pipeline/`: shared helpers for config/io/schema/policy/quality/training.
- `modelfiles/`: Ollama Modelfile templates.

## Quick Start

### 1) Collect data

```bash
python fine_tuning/collect_data.py --episodes 50 --output data/gold_standard_data.jsonl
```

### 2) Convert format

```bash
python fine_tuning/convert_data.py --input data/gold_standard_data.jsonl --output data/gold_standard_data_clean.jsonl
```

### 3) Validate dataset

```bash
python fine_tuning/validate_finetune_dataset.py data/gold_standard_data_clean.jsonl
```

### 4) Train and export merged model

```bash
python fine_tuning/train_dilu_ollama.py \
  --data-file data/gold_standard_data_clean.jsonl \
  --model-name unsloth/Llama-3.1-8B-Instruct \
  --model-family llama3 \
  --merged-model-dir fine_tuning/merged_models/dilu-llama3_1-8b-v1
```

### 5) Create Ollama model

Update `fine_tuning/modelfiles/dilu-llama3_1-8b-v1.Modelfile` so `FROM` points to your merged model folder, then run:

```bash
ollama create dilu-llama3_1-8b-v1 -f fine_tuning/modelfiles/dilu-llama3_1-8b-v1.Modelfile
```

### 6) GGUF-first packaging (recommended)

1. Build/clone `llama.cpp` locally and ensure `convert_hf_to_gguf.py` is available.
2. Convert merged model to GGUF (and optionally quantize), then generate an Ollama Modelfile:

```bash
python fine_tuning/build_gguf.py \
  --llama-cpp-dir C:/dev/llama.cpp \
  --hf-model-dir fine_tuning/merged_models/dilu-llama3_1-8b-v1 \
  --name dilu-llama3_1-8b-v1 \
  --outtype f16 \
  --quantize Q4_K_M
```

This creates:
- `fine_tuning/gguf/dilu-llama3_1-8b-v1.f16.gguf`
- `fine_tuning/gguf/dilu-llama3_1-8b-v1.Q4_K_M.gguf` (if quantized)
- `fine_tuning/gguf/dilu-llama3_1-8b-v1.gguf.Modelfile`

3. Create an Ollama model from generated GGUF Modelfile:

```bash
ollama create dilu-llama3_1-8b-v1-gguf -f fine_tuning/gguf/dilu-llama3_1-8b-v1.gguf.Modelfile
```

Or do step 3 automatically:

```bash
python fine_tuning/build_gguf.py \
  --llama-cpp-dir C:/dev/llama.cpp \
  --hf-model-dir fine_tuning/merged_models/dilu-llama3_1-8b-v1 \
  --name dilu-llama3_1-8b-v1 \
  --outtype f16 \
  --quantize Q4_K_M \
  --create-ollama \
  --ollama-model dilu-llama3_1-8b-v1-gguf
```

## Full Pipeline Command

Run all stages:

```bash
python fine_tuning/run_pipeline.py --all
```

Run only training on an existing cleaned dataset:

```bash
python fine_tuning/run_pipeline.py --train --clean-output data/gold_standard_data_clean.jsonl
```

## Notes

- `config.yaml` controls simulation settings used during data collection.
- Keep training datasets in `data/` (`gold_standard_data.jsonl`, `gold_standard_data_clean.jsonl`).
- For different base model families, set `--model-family` appropriately (`llama3`, `qwen`, `mistral`, or `auto`).
- Recommended deployment path in this repo: merged export -> GGUF -> `ollama create`.
