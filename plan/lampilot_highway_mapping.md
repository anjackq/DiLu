# LaMPilot Highway Mapping Review

## Scope

This note compares the **original upstream LaMPilot highway benchmark path** against the **local DiLu-Ollama `lampilot_highway_v1` implementation**.

Scope limits for this note:

- highway-style scenarios only
- no merge / intersection / pull-over analysis
- no paper-writing discussion
- focus on implementation mapping and missing pieces

Upstream reference used for this review:

- Repository: `https://github.com/PurdueDigitalTwin/LaMPilot`
- Local clone used for inspection: `temp/LaMPilot_upstream`
- Upstream commit inspected: `ba4564e28a32935f2a5de7b1b85bed7289fcebdb`

## Executive Summary

The local `lampilot_highway_v1` implementation is a **highway-only benchmark adaptation**, not a faithful local port of upstream LaMPilot.

What it preserves:

- the task-family idea of highway instruction following
- benchmark-style evaluation rather than pure reward logging
- safety / smoothness / time-efficiency style scoring
- lane change and overtaking as explicit task categories

What it changes:

- replaces the upstream language-to-code execution stack with the existing DiLu direct-action loop
- compresses the benchmark from a large config-driven dataset into a fixed 40-case JSON case set
- reimplements task evaluators inside a generic benchmark runtime instead of keeping separate evaluator classes
- adds local-runtime validity gating, which is central in DiLu-Ollama but not in upstream LaMPilot

## Benchmark Shape Comparison

### Upstream highway slice

- Upstream benchmark family: `DbLv1`
- Full upstream benchmark size: `32` config files, `4,900` total test items
- Highway-relevant task groups include:
  - absolute speed control
  - relative speed control
  - absolute following distance
  - relative following distance
  - left lane change
  - right lane change
  - left overtake
  - right overtake

### Local highway slice

- Local benchmark family: `lampilot_highway_v1`
- Local size: `40` total cases
- Categories:
  - `speed_increase` x `5`
  - `speed_decrease` x `5`
  - `follow_gap_increase` x `5`
  - `follow_gap_decrease` x `5`
  - `lane_change_left` x `5`
  - `lane_change_right` x `5`
  - `overtake_left` x `5`
  - `overtake_right` x `5`

### Interpretation

The local benchmark is best understood as a **compact highway-only protocol derived from upstream task families**, not as the upstream benchmark itself.

## File-By-File Mapping

### 1. Dataset / benchmark inventory

#### Upstream

- `temp/LaMPilot_upstream/projects/lampilot/dt/dbl.py`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/config_list.txt`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/*.json`

#### Local counterpart

- `benchmarks/lampilot_highway_v1/cases.json`
- `dilu/runtime/task_benchmark.py`

#### Mapping

- Upstream stores benchmark structure as:
  - many config files
  - each config file has `commands`
  - each config file has `samples`
  - dataset items are the cross-product of commands and samples
- Local stores benchmark structure as:
  - one case-set JSON
  - each row is already a concrete benchmark case
  - each case has a single fixed instruction
  - no command/sample cross-product exists

#### Status

- **Adapted, not equivalent**

#### What is missing locally

- command paraphrase coverage from upstream
- explicit config-per-task-file benchmark organization
- dataset object equivalent to `DbLv1Dataset`
- support for `commands x samples` expansion

### 2. Highway benchmark task configs

#### Upstream highway task files

- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/inc_abs_speed28.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/inc_abs_speed30.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/inc_abs_speed32.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/dec_abs_speed10.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/dec_abs_speed12.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/dec_abs_speed14.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/inc_rel_speed3.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/inc_rel_speed5.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/inc_rel_speed7.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/dec_rel_speed4.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/dec_rel_speed6.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/dec_rel_speed8.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/inc_abs_dis65.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/inc_abs_dis70.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/inc_abs_dis75.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/dec_abs_dis15.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/dec_abs_dis20.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/dec_abs_dis25.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/inc_rel_dis8.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/inc_rel_dis10.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/inc_rel_dis12.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/dec_rel_dis13.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/dec_rel_dis15.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/dec_rel_dis17.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/left_lc.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/right_lc.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/left_overtake.json`
- `temp/LaMPilot_upstream/projects/lampilot/configs/DbLv1/right_overtake.json`

#### Local counterpart

- `benchmarks/lampilot_highway_v1/cases.json`

#### Mapping

- Upstream highway tasks are split by:
  - absolute vs relative
  - speed vs distance
  - left/right lane or overtake direction
  - many command paraphrases per task
  - many environment samples per task
- Local tasks are collapsed into:
  - semantic categories only
  - one instruction per concrete case
  - fixed case count per category
  - generic `success_criteria` schema

#### Status

- **Heavily compressed**

#### What is missing locally

- absolute vs relative task distinction
- the richer set of target values used upstream
- the original task naming vocabulary
- paraphrase diversity for the same task
- per-task config modularity

### 3. Highway evaluator base class

#### Upstream

- `temp/LaMPilot_upstream/projects/lampilot/evaluator/base.py`

#### Local counterpart

- `dilu/runtime/task_benchmark.py`
- `evaluate_models_ollama.py`

#### Mapping

- Upstream `DbLEvaluator` owns:
  - env creation
  - vehicle warm-up
  - per-step queueing
  - TTC/speed/time score calculation
  - context generation for prompting
- Local splits these responsibilities:
  - env creation and rollout loop: `evaluate_models_ollama.py`
  - case validation and task scoring: `task_benchmark.py`
  - per-step traffic metrics: `extract_step_traffic_metrics()` in `evaluate_models_ollama.py`

#### Status

- **Functionally related, structurally different**

#### What is missing locally

- a dedicated benchmark evaluator abstraction comparable to `DbLEvaluator`
- benchmark-specific context generation layer tied to the benchmark object
- a single object that owns benchmark execution from env init through scoring

### 4. ACC / speed / following-distance evaluators

#### Upstream

- `temp/LaMPilot_upstream/projects/lampilot/evaluator/acc.py`

#### Local counterpart

- `dilu/runtime/task_benchmark.py`

#### Mapping

- Upstream uses dedicated evaluator classes:
  - `ACCEvalbySpeed`
  - `ACCEvalbyDistance`
- Local encodes those task types via generic success criteria:
  - `speed_band`
  - `front_gap_band`

#### Status

- **Concept preserved, implementation replaced**

#### What is missing locally

- explicit absolute vs relative ACC task semantics
- evaluator-class-level separation between speed-following and distance-following logic
- upstream-style temporal success windows tied to evaluator class semantics rather than generic `hold_steps`

### 5. Lane change evaluator

#### Upstream

- `temp/LaMPilot_upstream/projects/lampilot/evaluator/lane_change.py`

#### Local counterpart

- `dilu/runtime/task_benchmark.py`

#### Mapping

- Upstream success condition:
  - lane changed to target lane
  - vehicle heading aligned with lane heading
- Local success condition:
  - lane rank equals target lane rank
  - held for required steps

#### Status

- **Simplified**

#### What is missing locally

- heading-alignment check after lane change
- explicit lane geometry confirmation at completion
- a task-specific lane-change evaluator object

### 6. Overtake evaluator

#### Upstream

- `temp/LaMPilot_upstream/projects/lampilot/evaluator/overtake.py`

#### Local counterpart

- `dilu/runtime/task_benchmark.py`

#### Mapping

- Upstream overtake success condition:
  - ego reaches target lane
  - front vehicle is passed by more than `2 * vehicle length`
- Local overtake success condition:
  - required side lane was visited
  - ego position exceeds tracked front vehicle by `pass_margin_m`

#### Status

- **Semantically similar, re-authored**

#### What is missing locally

- vehicle-length-based pass criterion from upstream
- explicit target-lane completion requirement in the final success condition
- a dedicated overtake evaluator class

### 7. Vehicle digital twin / control layer

#### Upstream

- `temp/LaMPilot_upstream/projects/lampilot/dt/vehicle_dt.py`

#### Local counterpart

- no direct equivalent
- nearest local runtime surface is the DiLu evaluation loop in `evaluate_models_ollama.py`

#### Mapping

- Upstream has a `VehicleDigitalTwin` abstraction that:
  - executes generated policies
  - exposes high-level control targets
  - bridges planner output and continuous vehicle control
- Local runtime does not expose a digital twin abstraction here.
- Local loop produces a discrete action id `0..4` and steps the env directly.

#### Status

- **Missing**

#### What is missing locally

- a local digital twin abstraction
- target-lane / target-speed stateful control interface
- executable policy layer between language output and actuation

### 8. Prompt-time driving APIs

#### Upstream

- `temp/LaMPilot_upstream/projects/lampilot/prompts/apis.py`

#### Local counterpart

- no direct equivalent
- the local model is forced to output only a discrete action token in `evaluate_models_ollama.py`

#### Mapping

- Upstream prompt APIs include:
  - perception APIs
  - control APIs
  - route APIs
  - passenger-facing `say()`
- Local DiLu-Ollama prompt contract is much narrower:
  - output one action id only
  - no executable API surface is exposed to the model

#### Status

- **Missing**

#### What is missing locally

- `set_target_speed()`
- `set_target_lane()`
- `autopilot()`
- `detect_front_vehicle_in()`
- `get_left_lane()` / `get_right_lane()`
- route APIs such as turn control
- the entire “LLM writes policy using primitives” interface

### 9. Highway env configuration

#### Upstream

- highway env config is embedded per task file in `projects/lampilot/configs/DbLv1/*.json`

#### Local counterpart

- `dilu/runtime/highway_env_config.py`
- `benchmarks/lampilot_highway_v1/cases.json`

#### Mapping

- Upstream config files define task-local env specs directly.
- Local runtime resolves the env centrally and then applies case-local overrides.
- Local default highway target is:
  - `highway-fast-v0`
  - stop-capable discrete speeds `[0, 5, 10, 15, 20, 25, 30]`

#### Status

- **Reorganized**

#### What is missing locally

- per-task file-local env definitions
- direct preservation of upstream highway env variants such as `ramp-merge-v0` for the highway slice
- the upstream environment layouts tied to the original benchmark samples

### 10. Benchmark execution entrypoints

#### Upstream

- `temp/LaMPilot_upstream/projects/lampilot/test_icl.py`
- `temp/LaMPilot_upstream/projects/lampilot/test_hf.py`
- `temp/LaMPilot_upstream/projects/lampilot/demo.py`

#### Local counterpart

- `evaluate_models_ollama.py`
- `benchmark_energy_latency.py`

#### Mapping

- Upstream has distinct entrypoints for:
  - demo
  - code generation evaluation
  - human-feedback evaluation
- Local uses one main evaluation CLI plus a compatibility shim for energy benchmarking.

#### Status

- **Compressed**

#### What is missing locally

- separate benchmark modes for zero-shot / few-shot code generation
- human-feedback benchmark workflow
- direct execution path for generated driving programs

## Highway-Only Missing Components

If the goal were a closer **highway-only faithful reimplementation** of upstream LaMPilot, the main missing pieces are:

### Missing benchmark content

- the original upstream highway task inventory
- absolute and relative variants as distinct benchmark tasks
- paraphrase-heavy command sets per task
- sample-command cross-product dataset generation

### Missing evaluator fidelity

- dedicated ACC-by-speed evaluator
- dedicated ACC-by-distance evaluator
- lane-change heading alignment checks
- overtake success semantics closer to upstream

### Missing framework layers

- digital twin object for policy execution
- LLM-generated code execution path
- prompt API surface for control and perception primitives
- human-feedback / policy-repository workflow

### Missing benchmark identity controls

- no explicit benchmark fingerprint matching upstream task-set semantics
- local benchmark name may overstate fidelity to upstream `LaMPilot-Bench`

## What the Local Version Adds Instead

The local implementation is not merely smaller. It also introduces capabilities that are central to DiLu-Ollama and not the core of upstream LaMPilot:

- prevalidation of all benchmark cases before model evaluation
- direct local-runtime benchmarking under Ollama
- timeout / fallback / quarantine validity gating
- benchmark invalidity as an explicit reported outcome
- energy / latency instrumentation integrated with benchmark reporting

## Recommended Naming Interpretation

For the current repo state, the most accurate internal description is:

- `LaMPilot-inspired highway benchmark adaptation`

Less accurate descriptions are:

- `local implementation of LaMPilot`
- `faithful LaMPilot benchmark port`

## Suggested Next Follow-Up

If we continue from this note, the next useful step is:

1. decide whether the local name `lampilot_highway_v1` should stay as-is or be renamed to something more explicit
2. decide whether we want:
   - naming cleanup only
   - benchmark-fidelity improvements
   - or a true upstream-style highway reimplementation
