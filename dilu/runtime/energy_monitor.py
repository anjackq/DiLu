import json
import math
import os
import platform
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil


TOKEN_COUNT_METHOD = "whitespace_estimate"


def estimate_generated_tokens(text: Optional[str]) -> int:
    raw = str(text or "").strip()
    if not raw:
        return 0
    return len(re.findall(r"\S+", raw))


def ollama_version() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["ollama", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def system_hardware_snapshot() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "memory_total_bytes": int(vm.total),
        "memory_available_bytes": int(vm.available),
        "ollama_version": ollama_version(),
    }


def load_idle_calibration(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Idle calibration artifact must be a JSON object.")
    if payload.get("avg_idle_power_w") is None:
        raise ValueError("Idle calibration artifact missing `avg_idle_power_w`.")
    return payload


def save_idle_calibration(path: str, artifact: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    return os.path.abspath(path)


def enrich_episode_energy_metrics(
    episode: Dict[str, Any],
    *,
    energy_mode: str,
    raw_energy_j: Optional[float],
    idle_power_w: Optional[float],
    avg_power_w: Optional[float],
    peak_power_w: Optional[float],
    prompt_tokens_total: int,
    completion_tokens_total: int,
    total_tokens: int,
    token_count_method: str,
    token_usage_source: str,
    latency_to_first_action_sec: Optional[float],
) -> Dict[str, Any]:
    runtime_sec = float(episode.get("episode_runtime_sec") or 0.0)
    decisions = int(episode.get("decisions_made") or 0)
    tokens_generated_total = int(completion_tokens_total or 0)
    baseline_energy_j = None
    if idle_power_w is not None and runtime_sec > 0:
        baseline_energy_j = float(idle_power_w) * runtime_sec

    net_energy_j = raw_energy_j
    if raw_energy_j is not None and baseline_energy_j is not None:
        net_energy_j = max(0.0, float(raw_energy_j) - float(baseline_energy_j))

    if avg_power_w is None and raw_energy_j is not None and runtime_sec > 0:
        avg_power_w = float(raw_energy_j) / runtime_sec

    energy_per_decision_j = None
    if net_energy_j is not None and decisions > 0:
        energy_per_decision_j = float(net_energy_j) / float(decisions)

    energy_per_token_j = None
    if net_energy_j is not None and tokens_generated_total > 0:
        energy_per_token_j = float(net_energy_j) / float(tokens_generated_total)

    tokens_per_second = None
    if runtime_sec > 0 and tokens_generated_total > 0:
        tokens_per_second = float(tokens_generated_total) / runtime_sec

    enriched = dict(episode)
    enriched.update(
        {
            "energy_mode": energy_mode,
            "raw_energy_j": round(float(raw_energy_j), 6) if raw_energy_j is not None else None,
            "idle_baseline_energy_j": round(float(baseline_energy_j), 6) if baseline_energy_j is not None else None,
            "net_energy_j": round(float(net_energy_j), 6) if net_energy_j is not None else None,
            "avg_power_w": round(float(avg_power_w), 6) if avg_power_w is not None else None,
            "peak_power_w": round(float(peak_power_w), 6) if peak_power_w is not None else None,
            "energy_per_decision_j": round(float(energy_per_decision_j), 6) if energy_per_decision_j is not None else None,
            "energy_per_token_j": round(float(energy_per_token_j), 6) if energy_per_token_j is not None else None,
            "prompt_tokens_total": int(prompt_tokens_total or 0),
            "completion_tokens_total": int(completion_tokens_total or 0),
            "total_tokens": int(total_tokens or 0),
            "tokens_generated_total": int(tokens_generated_total),
            "tokens_per_second": round(float(tokens_per_second), 6) if tokens_per_second is not None else None,
            "latency_to_first_action_sec": round(float(latency_to_first_action_sec), 6) if latency_to_first_action_sec is not None else None,
            "token_count_method": str(token_count_method or TOKEN_COUNT_METHOD),
            "token_usage_source": str(token_usage_source or "estimate_fallback"),
        }
    )
    return enriched


def summarize_energy_latency_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not episodes:
        return {}

    def _mean(key: str) -> Optional[float]:
        values = [
            float(item.get(key))
            for item in episodes
            if item.get(key) is not None
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 6)

    return {
        "energy_mode": str(episodes[0].get("energy_mode") or "latency_only"),
        "episode_runtime_sec_mean": _mean("episode_runtime_sec"),
        "decision_latency_ms_avg_mean": _mean("decision_latency_ms_avg"),
        "p95_decision_latency_sec_mean": _mean("p95_decision_latency_sec"),
        "latency_to_first_action_sec_mean": _mean("latency_to_first_action_sec"),
        "raw_energy_j_mean": _mean("raw_energy_j"),
        "idle_baseline_energy_j_mean": _mean("idle_baseline_energy_j"),
        "net_energy_j_mean": _mean("net_energy_j"),
        "avg_power_w_mean": _mean("avg_power_w"),
        "peak_power_w_mean": _mean("peak_power_w"),
        "energy_per_decision_j_mean": _mean("energy_per_decision_j"),
        "energy_per_token_j_mean": _mean("energy_per_token_j"),
        "prompt_tokens_total_mean": _mean("prompt_tokens_total"),
        "completion_tokens_total_mean": _mean("completion_tokens_total"),
        "total_tokens_mean": _mean("total_tokens"),
        "tokens_generated_total_mean": _mean("tokens_generated_total"),
        "tokens_per_second_mean": _mean("tokens_per_second"),
        "token_count_method": (
            str(episodes[0].get("token_count_method"))
            if len({str(item.get("token_count_method") or "") for item in episodes}) == 1
            else "mixed"
        ),
        "token_usage_source": (
            str(episodes[0].get("token_usage_source"))
            if len({str(item.get("token_usage_source") or "") for item in episodes}) == 1
            else "mixed"
        ),
    }


@dataclass
class _PowerSample:
    wall_time_sec: float
    power_mean_w: float
    power_max_w: float
    energy_j: float


class BaseEnergyMonitor:
    mode = "latency_only"

    def __init__(self, idle_power_w: Optional[float] = None):
        self.idle_power_w = float(idle_power_w) if idle_power_w is not None else None

    def start_episode(self, episode_meta: Optional[Dict[str, Any]] = None) -> None:
        return None

    def stop_episode(self) -> Dict[str, Any]:
        return {
            "raw_energy_j": None,
            "avg_power_w": None,
            "peak_power_w": None,
            "measurement_meta": {},
        }

    def close(self) -> None:
        return None

    def calibrate_idle(self, duration_sec: float, output_path: str) -> str:
        raise RuntimeError(f"Idle calibration is not supported for energy mode `{self.mode}`.")


class LatencyOnlyEnergyMonitor(BaseEnergyMonitor):
    mode = "latency_only"


class JoulescopeHardwareEnergyMonitor(BaseEnergyMonitor):
    mode = "joulescope_hw"

    def __init__(self, idle_power_w: Optional[float] = None):
        super().__init__(idle_power_w=idle_power_w)
        try:
            import joulescope  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Energy mode `joulescope_hw` requires the optional `joulescope` Python package."
            ) from exc
        self._joulescope = joulescope
        self._ctx = joulescope.scan_require_one(config="auto")
        self._device = self._ctx.__enter__()
        self._lock = threading.Lock()
        self._active = False
        self._samples: List[_PowerSample] = []
        self._device.statistics_callback_register(self._statistics_callback, source="sensor")
        self._device.start()

    def _statistics_callback(self, stats: Dict[str, Any]) -> None:
        power = (((stats.get("signals") or {}).get("power") or {}))
        power_mean = (((power.get("µ") or {}).get("value")))
        power_max = (((power.get("max") or {}).get("value")))
        energy = ((((stats.get("accumulators") or {}).get("energy") or {}).get("value")))
        if power_mean is None or energy is None:
            return
        sample = _PowerSample(
            wall_time_sec=time.time(),
            power_mean_w=float(power_mean),
            power_max_w=float(power_max if power_max is not None else power_mean),
            energy_j=float(energy),
        )
        with self._lock:
            if self._active:
                self._samples.append(sample)

    def start_episode(self, episode_meta: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._samples = []
            self._active = True
        self._device.statistics_accumulators_clear()

    def stop_episode(self) -> Dict[str, Any]:
        time.sleep(0.15)
        with self._lock:
            self._active = False
            samples = list(self._samples)
        raw_energy_j = None
        avg_power_w = None
        peak_power_w = None
        if samples:
            raw_energy_j = max(0.0, float(samples[-1].energy_j))
            avg_power_w = float(sum(sample.power_mean_w for sample in samples) / len(samples))
            peak_power_w = float(max(sample.power_max_w for sample in samples))
        return {
            "raw_energy_j": raw_energy_j,
            "avg_power_w": avg_power_w,
            "peak_power_w": peak_power_w,
            "measurement_meta": {
                "sample_count": len(samples),
                "device_model": "Joulescope JS110",
            },
        }

    def calibrate_idle(self, duration_sec: float, output_path: str) -> str:
        duration_sec = max(1.0, float(duration_sec))
        with self._lock:
            self._samples = []
            self._active = True
        self._device.statistics_accumulators_clear()
        time.sleep(duration_sec)
        result = self.stop_episode()
        with self._lock:
            samples = list(self._samples)
        power_values = [sample.power_mean_w for sample in samples]
        artifact = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "energy_mode": self.mode,
            "duration_sec": round(duration_sec, 4),
            "avg_idle_power_w": round(float(statistics.mean(power_values)), 6) if power_values else None,
            "std_idle_power_w": round(float(statistics.pstdev(power_values)), 6) if len(power_values) > 1 else 0.0,
            "sample_count": len(power_values),
            "raw_energy_j": round(float(result["raw_energy_j"]), 6) if result.get("raw_energy_j") is not None else None,
            "device_model": "Joulescope JS110",
            "power_sample_hz_estimate": round(float(len(power_values) / duration_sec), 4) if duration_sec > 0 else None,
        }
        return save_idle_calibration(output_path, artifact)

    def close(self) -> None:
        try:
            self._device.statistics_callback_unregister(self._statistics_callback, source="sensor")
        except Exception:
            pass
        try:
            self._device.stop()
        except Exception:
            pass
        try:
            self._ctx.__exit__(None, None, None)
        except Exception:
            pass


def create_energy_monitor(
    energy_mode: str,
    *,
    idle_calibration: Optional[Dict[str, Any]] = None,
) -> BaseEnergyMonitor:
    normalized = str(energy_mode or "latency_only").strip().lower()
    idle_power_w = None
    if idle_calibration is not None and idle_calibration.get("avg_idle_power_w") is not None:
        idle_power_w = float(idle_calibration["avg_idle_power_w"])
    if normalized in {"none", "latency_only"}:
        return LatencyOnlyEnergyMonitor(idle_power_w=idle_power_w)
    if normalized == "joulescope_hw":
        return JoulescopeHardwareEnergyMonitor(idle_power_w=idle_power_w)
    raise ValueError(f"Unsupported energy mode: {energy_mode}")


def build_energy_tradeoff_summary(aggregates: List[Dict[str, Any]]) -> Dict[str, Any]:
    points = []
    for row in aggregates:
        points.append(
            {
                "model": row.get("model"),
                "net_energy_j_mean": row.get("net_energy_j_mean"),
                "energy_per_decision_j_mean": row.get("energy_per_decision_j_mean"),
                "energy_per_token_j_mean": row.get("energy_per_token_j_mean"),
                "tokens_per_second_mean": row.get("tokens_per_second_mean"),
                "decision_latency_ms_avg_mean": row.get("decision_latency_ms_avg_mean"),
                "p95_decision_latency_sec_mean": row.get("p95_decision_latency_sec_mean"),
                "crash_rate": row.get("crash_rate"),
                "task_completion_rate": row.get("task_completion_rate"),
                "driving_score": row.get("driving_score"),
                "driving_score_v2": row.get("driving_score_v2"),
            }
        )
    return {
        "headline_task_metric": "driving_score_v2",
        "efficiency_metrics_reported": True,
        "points": points,
        "pareto_objectives": {
            "minimize": [
                "net_energy_j_mean",
                "energy_per_decision_j_mean",
                "decision_latency_ms_avg_mean",
                "crash_rate",
            ],
            "maximize": [
                "driving_score_v2",
                "tokens_per_second_mean",
                "task_completion_rate",
                "driving_score",
            ],
        },
    }
