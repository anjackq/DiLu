## Enable Gemini API for DiLu Simulation Runtime

### Summary
Add first-class Gemini support (native SDK path) so `run_dilu_ollama.py` and related tools can run with Google Gemini models using `GEMINI_API_KEY`, while keeping existing `azure` / `openai` / `ollama` behavior unchanged.

Locked choices:
- Integration path: **native Gemini SDK** (`langchain-google-genai`)
- Embeddings strategy under Gemini mode: **keep existing embedding provider flow** (no Gemini embeddings in this phase)

---

## Goals and Success Criteria
1. User can set `OPENAI_API_TYPE: "gemini"` in `config.yaml` and run simulation successfully.
2. Driver agent and reflection agent both work with Gemini chat models.
3. Existing providers remain backward-compatible.
4. Memory retrieval still works by using the existing embedding provider settings.
5. README/config examples clearly document Gemini setup and commands.

---

## Scope

### In scope
1. Runtime provider wiring for `gemini`.
2. Driver/reflection LLM initialization branches for Gemini.
3. Config schema extension for Gemini keys.
4. User-facing docs and validation messages.
5. Basic smoke-test checklist.

### Out of scope
1. Gemini embeddings integration.
2. Fine-tuning changes.
3. Metric schema changes.
4. Judge-model/alignment scoring changes.

---

## Implementation Plan

### 1) Dependencies
1. Add `langchain-google-genai` to [`requirements.txt`](C:\Users\WiCon\Desktop\DiLu-Ollama\requirements.txt).
2. Keep existing LangChain/OpenAI deps unchanged.

### 2) Config surface and defaults
1. Extend [`config.example.yaml`](C:\Users\WiCon\Desktop\DiLu-Ollama\config.example.yaml):
   - `OPENAI_API_TYPE: 'gemini'` (document as supported value)
   - `GEMINI_API_KEY: ''`
   - `GEMINI_CHAT_MODEL: 'gemini-2.0-flash'` (or chosen default)
   - `GEMINI_REFLECTION_MODEL: null` (fallback to `GEMINI_CHAT_MODEL`)
2. Keep existing embedding settings unchanged (`OLLAMA_EMBED_MODEL`, etc.).
3. Keep `config.yaml` usage behavior unchanged (runtime still reads `config.yaml`).

### 3) Runtime env wiring
1. Update [`dilu/runtime/llm_env.py`](C:\Users\WiCon\Desktop\DiLu-Ollama\dilu\runtime\llm_env.py):
   - `_pick_model` handles `api_type == "gemini"` via `GEMINI_CHAT_MODEL`.
   - `configure_runtime_env` new `gemini` branch:
     - validate `GEMINI_API_KEY` and `GEMINI_CHAT_MODEL`.
     - set:
       - `OPENAI_API_TYPE=gemini`
       - `GEMINI_API_KEY`
       - `GEMINI_CHAT_MODEL`
       - optional `GEMINI_REFLECTION_MODEL`
     - return selected model.
2. Keep existing branches unchanged and error on unsupported type.

### 4) Driver agent Gemini client
1. Update [`dilu/driver_agent/driverAgent.py`](C:\Users\WiCon\Desktop\DiLu-Ollama\dilu\driver_agent\driverAgent.py):
   - import `ChatGoogleGenerativeAI`.
   - add `elif oai_api_type == "gemini":` branch in constructor:
     - model from `GEMINI_CHAT_MODEL`
     - key from `GEMINI_API_KEY`
     - temperature/request timeout/max tokens mapped to current behavior
     - keep streaming behavior compatible with existing response collection loop.
2. Keep action parsing/fallback logic unchanged.

### 5) Reflection agent Gemini client
1. Update [`dilu/driver_agent/reflectionAgent.py`](C:\Users\WiCon\Desktop\DiLu-Ollama\dilu\driver_agent\reflectionAgent.py):
   - add Gemini branch with `ChatGoogleGenerativeAI`.
   - default reflection model:
     - `GEMINI_REFLECTION_MODEL` if set, else `GEMINI_CHAT_MODEL`.
2. Keep reflection prompt and output parsing unchanged.

### 6) Memory/embedding behavior under Gemini
1. Update [`dilu/driver_agent/vectorStore.py`](C:\Users\WiCon\Desktop\DiLu-Ollama\dilu\driver_agent\vectorStore.py):
   - add explicit `OPENAI_API_TYPE == "gemini"` branch that **reuses existing embedding provider path**.
2. Implementation rule for this phase:
   - If `OPENAI_API_BASE`/`OPENAI_API_KEY` (OpenAI-compatible embedding settings) are present, use those.
   - Else fallback to Ollama embedding env (`OLLAMA_API_BASE`, `OLLAMA_API_KEY`, `OLLAMA_EMBED_MODEL`) and emit clear log line.
   - If neither available, raise actionable error explaining required embedding settings in Gemini mode.
3. Do not introduce Gemini embedding model calls.

### 7) Script messaging consistency
1. Update provider info messages in:
   - [`run_dilu_ollama.py`](C:\Users\WiCon\Desktop\DiLu-Ollama\run_dilu_ollama.py)
   - [`memory_check.py`](C:\Users\WiCon\Desktop\DiLu-Ollama\memory_check.py)
   - [`visualize_results.py`](C:\Users\WiCon\Desktop\DiLu-Ollama\visualize_results.py)
2. Ensure no Ollama-specific message is printed for Gemini runs.
3. Keep results schema unchanged except existing provider value now may be `"gemini"`.

### 8) Documentation updates
1. Update root [`README.md`](C:\Users\WiCon\Desktop\DiLu-Ollama\README.md):
   - add Gemini section:
     - required config keys
     - example run command
     - note: embeddings remain non-Gemini in this phase.
2. Update troubleshooting:
   - invalid key / model not found / embedding backend missing.

---

## Public Interface Changes

### Config keys (additive)
1. `OPENAI_API_TYPE`: now supports `"gemini"`.
2. `GEMINI_API_KEY` (new).
3. `GEMINI_CHAT_MODEL` (new).
4. `GEMINI_REFLECTION_MODEL` (new, optional).

### Runtime environment variables (additive)
1. `GEMINI_API_KEY`
2. `GEMINI_CHAT_MODEL`
3. `GEMINI_REFLECTION_MODEL` (optional)

No existing key removal/rename.

---

## Validation and Test Plan

### A) Static checks
1. Import and init checks for `ChatGoogleGenerativeAI`.
2. `configure_runtime_env` returns selected model for `gemini` and throws clear errors on missing key/model.

### B) Smoke runs
1. Single-episode simulation with:
   - `OPENAI_API_TYPE=gemini`
   - valid Gemini key/model
   - existing embedding backend configured.
2. Reflection on/off both paths.
3. `memory_check.py` works in Gemini mode with configured embedding backend.

### C) Regression checks
1. One short run each for `ollama` and `openai` mode still works.
2. Existing output artifacts (`log.txt`, `.db`, videos, run metrics) unchanged in structure.
3. `evaluate_models_ollama.py` still runs for non-Gemini paths.

### D) Failure-mode checks
1. Missing `GEMINI_API_KEY` -> clear startup error.
2. Missing embedding backend in Gemini mode -> actionable error in `DrivingMemory`.
3. Invalid model name -> provider/API exception surfaced with context.

---

## Risks and Mitigations
1. LangChain Gemini API differences (streaming token format):
   - Mitigation: keep response assembly centralized and test stream/non-stream fallback.
2. Embedding backend mismatch when switching providers:
   - Mitigation: explicit logs + strict validation + separate memory folder recommendation.
3. Provider-specific timeout/token kwargs:
   - Mitigation: map only supported kwargs for Gemini branch.

---

## Assumptions and Defaults
1. Gemini usage is via Google AI Studio API key (`GEMINI_API_KEY`), not Vertex AI.
2. Default Gemini model is `gemini-2.0-flash` unless overridden.
3. Embeddings remain on current non-Gemini backend in this phase.
4. Existing config file used by scripts is `config.yaml`; `config.example.yaml` is template only.
