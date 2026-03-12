import os
from typing import Any, Dict, Optional


def _pick_model(config: Dict[str, Any], api_type: str, chat_model_override: Optional[str]) -> Optional[str]:
    if chat_model_override:
        return str(chat_model_override)
    if api_type == "openai":
        model = config.get("OPENAI_CHAT_MODEL")
    elif api_type == "ollama":
        model = config.get("OLLAMA_CHAT_MODEL")
    elif api_type == "gemini":
        model = config.get("GEMINI_CHAT_MODEL")
    else:
        return None
    if model is None:
        return None
    model_str = str(model).strip()
    if not model_str or model_str.lower() == "none":
        return None
    return model_str


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _apply_output_runtime_controls(config: Dict[str, Any]) -> None:
    """
    Runtime output controls are configured here (outside policy resolution).
    """
    disable_streaming = _as_bool(config.get("eval_disable_streaming", True), default=True)
    disable_checker = _as_bool(config.get("eval_disable_checker_llm", True), default=True)
    try:
        max_output_tokens = int(config.get("eval_decision_max_output_tokens", 512))
    except Exception:
        max_output_tokens = 512
    max_output_tokens = max(1, max_output_tokens)

    os.environ["DILU_USE_STREAMING"] = "0" if disable_streaming else "1"
    os.environ["DILU_ENABLE_CHECKER_LLM"] = "0" if disable_checker else "1"
    os.environ["DILU_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)


def _resolve_quiet_mode(config: Dict[str, Any], mode: str, quiet_override: Optional[bool]) -> bool:
    if quiet_override is not None:
        return bool(quiet_override)
    mode_normalized = str(mode or "runtime").strip().lower()
    global_default = _as_bool(config.get("quiet_mode", False), default=False)
    mode_key = "eval_quiet_mode" if mode_normalized == "eval" else "runtime_quiet_mode"
    mode_value = config.get(mode_key)
    if mode_value is None:
        return global_default
    return _as_bool(mode_value, default=global_default)


def _resolve_progress_bar(config: Dict[str, Any], mode: str, progress_override: Optional[bool]) -> bool:
    if progress_override is not None:
        return bool(progress_override)
    mode_normalized = str(mode or "runtime").strip().lower()
    global_default = _as_bool(config.get("progress_bar", True), default=True)
    mode_key = "eval_progress_bar" if mode_normalized == "eval" else "runtime_progress_bar"
    mode_value = config.get(mode_key)
    if mode_value is None:
        return global_default
    return _as_bool(mode_value, default=global_default)


def configure_runtime_env(
    config: Dict[str, Any],
    chat_model_override: Optional[str] = None,
    mode: str = "runtime",
    quiet_override: Optional[bool] = None,
    progress_override: Optional[bool] = None,
) -> Optional[str]:
    """
    Configure provider-specific env vars used by DiLu runtime scripts.

    Returns the selected chat model for providers that require one.
    """
    api_type = str(config["OPENAI_API_TYPE"]).strip().lower()
    selected_model = _pick_model(config, api_type, chat_model_override)
    _apply_output_runtime_controls(config)
    os.environ["DILU_QUIET_MODE"] = "1" if _resolve_quiet_mode(config, mode, quiet_override) else "0"
    os.environ["DILU_PROGRESS_BAR"] = "1" if _resolve_progress_bar(config, mode, progress_override) else "0"

    if api_type == "azure":
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = str(config["AZURE_API_VERSION"])
        os.environ["OPENAI_API_BASE"] = str(config["AZURE_API_BASE"])
        os.environ["OPENAI_API_KEY"] = str(config["AZURE_API_KEY"])
        os.environ["AZURE_CHAT_DEPLOY_NAME"] = str(config["AZURE_CHAT_DEPLOY_NAME"])
        os.environ["AZURE_EMBED_DEPLOY_NAME"] = str(config["AZURE_EMBED_DEPLOY_NAME"])
        return selected_model

    if api_type == "openai":
        if not selected_model:
            raise ValueError("OPENAI_CHAT_MODEL must be set when OPENAI_API_TYPE is 'openai'.")
        os.environ["OPENAI_API_TYPE"] = "openai"
        os.environ["OPENAI_API_KEY"] = str(config["OPENAI_KEY"])
        os.environ["OPENAI_CHAT_MODEL"] = selected_model
        if config.get("OPENAI_REFLECTION_MODEL"):
            os.environ["OPENAI_REFLECTION_MODEL"] = str(config["OPENAI_REFLECTION_MODEL"])
        return selected_model

    if api_type == "ollama":
        if not selected_model:
            raise ValueError("OLLAMA_CHAT_MODEL must be set when OPENAI_API_TYPE is 'ollama'.")
        ollama_api_base = str(config.get("OLLAMA_API_BASE", "http://localhost:11434/v1"))
        ollama_api_key = str(config.get("OLLAMA_API_KEY", "ollama"))
        ollama_think_mode = str(config.get("OLLAMA_THINK_MODE", "auto"))
        ollama_use_native_chat = str(config.get("OLLAMA_USE_NATIVE_CHAT", True))
        ollama_native_chat_timeout_sec = str(config.get("OLLAMA_NATIVE_CHAT_TIMEOUT_SEC", 60))
        os.environ["OPENAI_API_TYPE"] = "ollama"
        os.environ["OLLAMA_API_BASE"] = ollama_api_base
        os.environ["OPENAI_BASE_URL"] = ollama_api_base
        # Keep legacy var for compatibility with some clients/tools.
        os.environ["OPENAI_API_BASE"] = ollama_api_base
        os.environ["OLLAMA_API_KEY"] = ollama_api_key
        os.environ["OLLAMA_CHAT_MODEL"] = selected_model
        os.environ["OLLAMA_EMBED_MODEL"] = str(config["OLLAMA_EMBED_MODEL"])
        os.environ["OLLAMA_THINK_MODE"] = ollama_think_mode
        os.environ["OLLAMA_USE_NATIVE_CHAT"] = ollama_use_native_chat
        os.environ["OLLAMA_NATIVE_CHAT_TIMEOUT_SEC"] = ollama_native_chat_timeout_sec
        if config.get("OLLAMA_REFLECTION_MODEL"):
            os.environ["OLLAMA_REFLECTION_MODEL"] = str(config["OLLAMA_REFLECTION_MODEL"])
        return selected_model

    if api_type == "gemini":
        if not selected_model:
            raise ValueError("GEMINI_CHAT_MODEL must be set when OPENAI_API_TYPE is 'gemini'.")
        gemini_api_key = str(config.get("GEMINI_API_KEY", "")).strip()
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set when OPENAI_API_TYPE is 'gemini'.")
        os.environ["OPENAI_API_TYPE"] = "gemini"
        os.environ["GEMINI_API_KEY"] = gemini_api_key
        os.environ["GEMINI_CHAT_MODEL"] = selected_model
        if config.get("GEMINI_REFLECTION_MODEL"):
            os.environ["GEMINI_REFLECTION_MODEL"] = str(config["GEMINI_REFLECTION_MODEL"])
        # Gemini mode reuses existing embedding backends (no native Gemini embeddings in this phase).
        # If Ollama embedding settings exist in config, expose them as env vars for DrivingMemory.
        if config.get("OLLAMA_API_BASE"):
            os.environ["OLLAMA_API_BASE"] = str(config["OLLAMA_API_BASE"])
        if config.get("OLLAMA_API_KEY"):
            os.environ["OLLAMA_API_KEY"] = str(config["OLLAMA_API_KEY"])
        if config.get("OLLAMA_EMBED_MODEL"):
            os.environ["OLLAMA_EMBED_MODEL"] = str(config["OLLAMA_EMBED_MODEL"])
        return selected_model

    raise ValueError(f"Unsupported OPENAI_API_TYPE: {config['OPENAI_API_TYPE']}")
