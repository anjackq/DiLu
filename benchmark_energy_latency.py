import sys
from typing import List, Optional

from rich import print

from evaluate_models_ollama import main as evaluate_main


DEFAULT_BENCHMARK_ENERGY_MODE = "latency_only"
DEFAULT_BENCHMARK_RESULTS_ROOT = "results/energy_benchmarks"
DEFAULT_BENCHMARK_EXPERIMENT_ID = "energy_latency_benchmark"


def _find_flag(argv: List[str], flag: str) -> int:
    try:
        return argv.index(flag)
    except ValueError:
        return -1


def _has_any_flag(argv: List[str], *flags: str) -> bool:
    return any(flag in argv for flag in flags)


def translate_benchmark_args_to_eval_argv(argv: Optional[List[str]] = None) -> List[str]:
    translated = list(sys.argv[1:] if argv is None else argv)

    energy_idx = _find_flag(translated, "--energy-mode")
    if energy_idx >= 0 and energy_idx + 1 < len(translated):
        if str(translated[energy_idx + 1]).strip().lower() == "none":
            translated[energy_idx + 1] = DEFAULT_BENCHMARK_ENERGY_MODE
    else:
        translated.extend(["--energy-mode", DEFAULT_BENCHMARK_ENERGY_MODE])

    if not _has_any_flag(translated, "--results-root", "--output-root"):
        translated.extend(["--results-root", DEFAULT_BENCHMARK_RESULTS_ROOT])

    if "--experiment-id" not in translated:
        translated.extend(["--experiment-id", DEFAULT_BENCHMARK_EXPERIMENT_ID])

    return translated


def main(argv: Optional[List[str]] = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    translated = translate_benchmark_args_to_eval_argv(raw_argv)
    if _has_any_flag(raw_argv, "-h", "--help"):
        print(
            "[yellow]Compatibility note:[/yellow] this shim forwards to "
            "`evaluate_models_ollama.py` and injects benchmark defaults: "
            f"`--energy-mode {DEFAULT_BENCHMARK_ENERGY_MODE}`, "
            f"`--results-root {DEFAULT_BENCHMARK_RESULTS_ROOT}`, "
            f"`--experiment-id {DEFAULT_BENCHMARK_EXPERIMENT_ID}`."
        )
    else:
        print(
            "[yellow]`benchmark_energy_latency.py` is now a compatibility shim. "
            "Use `evaluate_models_ollama.py --energy-mode ...` as the canonical entrypoint.[/yellow]"
        )
    evaluate_main(translated)


if __name__ == "__main__":
    main()
