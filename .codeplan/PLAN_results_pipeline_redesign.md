## Results Pipeline Redesign: Per-Experiment + Per-Model Output Structure

### Summary
Redesign result generation and comparison so every model has a dedicated folder, while preserving current scripts and compatibility.  
Adopt a **per-experiment root** layout, with:
- full run artifacts per model (videos, DB, logs, run metrics)
- global comparison report
- per-model comparison summaries copied from global report

This plan keeps existing workflows working, but adds deterministic foldering and manifest metadata for reproducibility.

---

## 1. Target Folder Structure (Locked)

Use this canonical layout:

```text
results/
  experiments/
    <experiment_id>/
      manifest.json
      compare/
        eval_compare_<timestamp>.json
        eval_compare_<timestamp>_plot.png
      models/
        <model_slug>/
          runs/
            <run_id>/
              log.txt
              highway_<episode>.db
              rl-video-episode-*.mp4
              run_metrics_<timestamp>.json
          eval/
            eval_summary_<timestamp>.json
            eval_episodes_<timestamp>.json
          plots/
            model_metrics_<timestamp>.png
```

### Naming rules
- `experiment_id`: explicit CLI/config or auto timestamp (`YYYYMMDD_HHMMSS`)
- `model_slug`: normalized from model tag (e.g., `deepseek-r1_14b`)
- `run_id`: `run_<timestamp>` unless overridden

---

## 2. Shared Path Utilities (New Core)

### File: `dilu/runtime/path_utils.py`
Add helper APIs (non-breaking additions):
1. `slugify_model_name(model_name: str) -> str`
2. `build_experiment_root(results_root: str, experiment_id: str) -> str`
3. `build_model_root(experiment_root: str, model_name: str) -> str`
4. `build_model_run_dir(experiment_root: str, model_name: str, run_id: str) -> str`
5. `ensure_experiment_layout(experiment_root, model_names: list[str]) -> dict`

All scripts use these helpers instead of hand-built paths.

---

## 3. `run_dilu_ollama.py` Changes

### Behavior
1. Resolve selected model via `configure_runtime_env(...)`.
2. Auto-build model run folder by default:
   - `results/experiments/<experiment_id>/models/<model_slug>/runs/<run_id>/`
3. Keep backward compatibility:
   - if `result_folder` explicit override is set, allow opt-out behavior via config/flag (see defaults section).
4. Write all existing run outputs into run folder:
   - videos, DBs, log, `run_metrics_*.json`.

### Report metadata additions
Append to run metrics root:
- `experiment_id`
- `experiment_root`
- `model_root`
- `run_id`
- `run_dir`

---

## 4. `evaluate_models_ollama.py` Changes

### New output behavior
For model list eval:
1. Create experiment root and model roots.
2. Save global compare report under:
   - `.../compare/eval_compare_<timestamp>.json`
3. Also write per-model extracts:
   - `.../models/<model_slug>/eval/eval_summary_<timestamp>.json`
   - `.../models/<model_slug>/eval/eval_episodes_<timestamp>.json`

### Per-model extract schema
- `model`
- `experiment_id`
- `source_compare_report`
- `aggregate` (single model aggregate block)
- `episodes` (per-seed episode list)
- `metrics_config`
- `created_at`

### CLI additions (non-breaking)
- `--experiment-id` (optional)
- `--results-root` (default `results/experiments`)
- `--output-root` optional override for compare folder
- keep existing `--output` behavior; if provided, still write there + keep new structured outputs (unless `--no-structured-output` is explicitly set)

---

## 5. `plot_eval_compare.py` Changes

### Input support
- Already supports global compare and single run metrics.
- Add optional `--emit-per-model`:
  - if input is global compare report, also emit one plot per model under each model’s `plots/` folder.

### Output defaults
- If input is inside experiment tree and `-o` not provided:
  - global plot goes next to input
  - per-model plots go to each model plot folder when enabled

---

## 6. Experiment Manifest (New)

### File: `results/experiments/<experiment_id>/manifest.json`
Track:
- `experiment_id`
- `created_at`
- `models`
- `config_path`
- `memory_path`
- `few_shot_num`
- `simulation_duration`
- file pointers:
  - latest run metrics per model
  - latest per-model eval summaries
  - latest global compare report
  - latest plots

Update manifest atomically after each run/eval operation.

---

## 7. Config & Interface Additions

### `config.example.yaml` additions
- `results_root: "results/experiments"`
- `experiment_id: null` (auto if null)
- `run_id: null` (auto if null)
- `result_folder_override: null` (explicit old-style folder override)

### Defaults chosen
- **Per-experiment root** (selected)
- **Global compare + per-model copies** (selected)
- **Auto result folder by model** (selected)

Compatibility default:
- if `result_folder_override` is present, use it exactly and log that structured mode was bypassed.

---

## 8. README Updates

Add a new section: “Experiment-Oriented Results Layout”
- folder tree
- example commands for:
  - single-model run
  - multi-model compare
  - plotting
- how to retrieve latest per-model summary quickly

---

## 9. Migration & Backward Compatibility

1. Existing commands continue to run unchanged.
2. Existing direct paths (`result_folder`, custom `--output`) still honored.
3. New layout is default unless explicit override is provided.
4. Existing report schema fields remain; only additive metadata fields are introduced.

---

## 10. Test Plan

### Unit tests (or script-level checks)
1. `slugify_model_name` normalization:
- `deepseek-r1:14b -> deepseek-r1_14b`
- names with `/`, spaces, uppercase handled safely.

2. Path resolver:
- correct creation of experiment/model/run directories.

3. Manifest update:
- idempotent updates, latest pointers refresh correctly.

### Integration scenarios
1. Single model `run_dilu_ollama.py`:
- videos, DB, log, run metrics saved under model run dir.

2. Multi-model `evaluate_models_ollama.py`:
- one global compare report + per-model summaries emitted.

3. Plotting:
- global plot from compare report
- per-model plots generated when requested.

4. Compatibility:
- explicit `result_folder_override` keeps legacy behavior.

### Acceptance criteria
- Every model has isolated artifacts under its own folder.
- Global compare remains available for cross-model analysis.
- Run/eval/plot outputs are discoverable via manifest without manual path hunting.
- No regression in reflection flow, video writing, or existing metrics keys.

---

## 11. Explicit Assumptions

1. Evaluation script remains metrics-only (no per-model video capture in eval path).
2. `results/experiments` is writable on target environments.
3. Existing report schema versioning remains `2.0` with additive metadata.
4. Single source of truth for path conventions is `dilu/runtime/path_utils.py`.
