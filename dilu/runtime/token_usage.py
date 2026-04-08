import copy
from typing import Any, Dict, Iterable, Optional


WHITESPACE_ESTIMATE_METHOD = "whitespace_estimate"
OLLAMA_NATIVE_USAGE_METHOD = "ollama_native_usage"
OLLAMA_OPENAI_USAGE_METHOD = "ollama_openai_usage"

NATIVE_API_SOURCE = "native_api"
OPENAI_COMPAT_SOURCE = "openai_compat"
ESTIMATE_FALLBACK_SOURCE = "estimate_fallback"


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _build_usage_record(
    *,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
    token_count_method: str,
    token_usage_source: str,
) -> Optional[Dict[str, Any]]:
    prompt = _as_int(prompt_tokens)
    completion = _as_int(completion_tokens)
    total = _as_int(total_tokens)
    if prompt is None and completion is None and total is None:
        return None
    prompt = max(0, int(prompt or 0))
    completion = max(0, int(completion or 0))
    total = max(0, int(total if total is not None else (prompt + completion)))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "token_count_method": token_count_method,
        "token_usage_source": token_usage_source,
    }


def build_token_usage_record_from_ollama_native_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    return _build_usage_record(
        prompt_tokens=payload.get("prompt_eval_count"),
        completion_tokens=payload.get("eval_count"),
        total_tokens=None,
        token_count_method=OLLAMA_NATIVE_USAGE_METHOD,
        token_usage_source=NATIVE_API_SOURCE,
    )


def build_token_usage_record_from_langchain_message(message: Any) -> Optional[Dict[str, Any]]:
    usage_metadata = getattr(message, "usage_metadata", None) or {}
    if isinstance(usage_metadata, dict) and usage_metadata:
        usage = _build_usage_record(
            prompt_tokens=usage_metadata.get("input_tokens"),
            completion_tokens=usage_metadata.get("output_tokens"),
            total_tokens=usage_metadata.get("total_tokens"),
            token_count_method=OLLAMA_OPENAI_USAGE_METHOD,
            token_usage_source=OPENAI_COMPAT_SOURCE,
        )
        if usage is not None:
            return usage

    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = None
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
    if isinstance(token_usage, dict):
        return _build_usage_record(
            prompt_tokens=token_usage.get("prompt_tokens"),
            completion_tokens=token_usage.get("completion_tokens"),
            total_tokens=token_usage.get("total_tokens"),
            token_count_method=OLLAMA_OPENAI_USAGE_METHOD,
            token_usage_source=OPENAI_COMPAT_SOURCE,
        )
    return None


def build_whitespace_estimate_token_usage(completion_tokens: int) -> Dict[str, Any]:
    completion = max(0, int(completion_tokens or 0))
    return {
        "prompt_tokens": 0,
        "completion_tokens": completion,
        "total_tokens": completion,
        "token_count_method": WHITESPACE_ESTIMATE_METHOD,
        "token_usage_source": ESTIMATE_FALLBACK_SOURCE,
    }


def combine_token_usage_records(*records: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    items = [copy.deepcopy(record) for record in records if isinstance(record, dict)]
    if not items:
        return None

    prompt_total = sum(max(0, int(item.get("prompt_tokens", 0) or 0)) for item in items)
    completion_total = sum(max(0, int(item.get("completion_tokens", 0) or 0)) for item in items)
    total_tokens = sum(max(0, int(item.get("total_tokens", 0) or 0)) for item in items)

    methods = sorted(
        {str(item.get("token_count_method") or "").strip() for item in items if item.get("token_count_method")}
    )
    sources = sorted(
        {str(item.get("token_usage_source") or "").strip() for item in items if item.get("token_usage_source")}
    )

    return {
        "prompt_tokens": int(prompt_total),
        "completion_tokens": int(completion_total),
        "total_tokens": int(total_tokens),
        "token_count_method": methods[0] if len(methods) == 1 else ("mixed" if methods else WHITESPACE_ESTIMATE_METHOD),
        "token_usage_source": sources[0] if len(sources) == 1 else ("mixed" if sources else ESTIMATE_FALLBACK_SOURCE),
    }


def aggregate_episode_token_usage(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    combined = combine_token_usage_records(*list(records))
    if combined is None:
        combined = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "token_count_method": WHITESPACE_ESTIMATE_METHOD,
            "token_usage_source": ESTIMATE_FALLBACK_SOURCE,
        }

    return {
        "prompt_tokens_total": int(combined["prompt_tokens"]),
        "completion_tokens_total": int(combined["completion_tokens"]),
        "total_tokens": int(combined["total_tokens"]),
        "tokens_generated_total": int(combined["completion_tokens"]),
        "token_count_method": str(combined["token_count_method"]),
        "token_usage_source": str(combined["token_usage_source"]),
    }
