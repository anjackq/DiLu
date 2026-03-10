import argparse
import subprocess
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end fine-tuning pipeline runner.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--collect", action="store_true", help="Run data collection stage.")
    parser.add_argument("--convert", action="store_true", help="Run data conversion stage.")
    parser.add_argument("--validate", action="store_true", help="Run dataset validation stage.")
    parser.add_argument("--train", action="store_true", help="Run training stage.")
    parser.add_argument("--gguf", action="store_true", help="Run GGUF conversion stage after training.")
    parser.add_argument("--all", action="store_true", help="Run all stages.")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes for collect stage.")
    parser.add_argument("--raw-output", default="data/gold_standard_data.jsonl", help="Raw JSONL output.")
    parser.add_argument("--clean-output", default="data/gold_standard_data_clean.jsonl", help="Clean JSONL output.")
    parser.add_argument("--model-name", default="unsloth/Llama-3.1-8B-Instruct", help="Training base model.")
    parser.add_argument("--model-family", default="auto", choices=["auto", "llama3", "qwen", "mistral", "deepseek", "phi"], help="Model family preset for training.")
    parser.add_argument("--merged-model-dir", default="fine_tuning/merged_models/dilu-llama3_1-8b-v1", help="Merged model output.")
    parser.add_argument("--llama-cpp-dir", default="", help="Optional llama.cpp dir for GGUF conversion (auto-detected if empty).")
    parser.add_argument("--gguf-output-dir", default="fine_tuning/gguf", help="Output directory for GGUF artifacts.")
    parser.add_argument("--gguf-name", default="", help="Optional output model name for GGUF files.")
    parser.add_argument("--gguf-outtype", default="f16", choices=["f16", "bf16", "f32", "q8_0", "auto"], help="GGUF outtype.")
    parser.add_argument("--gguf-quantize", default="", help="Optional quantization type, e.g. Q4_K_M.")
    parser.add_argument("--gguf-create-ollama", action="store_true", help="Run ollama create after GGUF build.")
    parser.add_argument("--ollama-model", default="", help="Ollama model name for --gguf-create-ollama.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_all = args.all or not any([args.collect, args.convert, args.validate, args.train, args.gguf])

    if args.collect or run_all:
        _run(
            [
                args.python,
                os.path.join(ROOT_DIR, "fine_tuning/collect_data.py"),
                "--episodes",
                str(args.episodes),
                "--output",
                args.raw_output,
            ]
        )

    if args.convert or run_all:
        _run(
            [
                args.python,
                os.path.join(ROOT_DIR, "fine_tuning/convert_data.py"),
                "--input",
                args.raw_output,
                "--output",
                args.clean_output,
            ]
        )

    if args.validate or run_all:
        _run([args.python, os.path.join(ROOT_DIR, "fine_tuning/validate_finetune_dataset.py"), args.clean_output])

    if args.train or run_all:
        _run(
            [
                args.python,
                os.path.join(ROOT_DIR, "fine_tuning/train_dilu_ollama.py"),
                "--data-file",
                args.clean_output,
                "--model-name",
                args.model_name,
                "--model-family",
                args.model_family,
                "--merged-model-dir",
                args.merged_model_dir,
            ]
        )

    if args.gguf:
        gguf_cmd = [
            args.python,
            os.path.join(ROOT_DIR, "fine_tuning/build_gguf.py"),
            "--hf-model-dir",
            args.merged_model_dir,
            "--output-dir",
            args.gguf_output_dir,
            "--outtype",
            args.gguf_outtype,
        ]
        if args.llama_cpp_dir.strip():
            gguf_cmd.extend(["--llama-cpp-dir", args.llama_cpp_dir])
        if args.gguf_name.strip():
            gguf_cmd.extend(["--name", args.gguf_name])
        if args.gguf_quantize.strip():
            gguf_cmd.extend(["--quantize", args.gguf_quantize])
        if args.gguf_create_ollama:
            gguf_cmd.append("--create-ollama")
            if args.ollama_model.strip():
                gguf_cmd.extend(["--ollama-model", args.ollama_model])

        _run(gguf_cmd)


if __name__ == "__main__":
    main()
