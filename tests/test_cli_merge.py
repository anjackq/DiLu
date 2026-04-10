import unittest
import os
import tempfile

import yaml

from benchmark_energy_latency import translate_benchmark_args_to_eval_argv
from dilu.runtime import resolve_simulation_env_bundle
from evaluate_models_ollama import _apply_measurement_runtime_overrides
from dilu.runtime import load_runtime_config


class _Args:
    def __init__(
        self,
        *,
        ollama_think_mode=None,
        ollama_use_native_chat=False,
        ollama_disable_native_chat=False,
    ):
        self.ollama_think_mode = ollama_think_mode
        self.ollama_use_native_chat = ollama_use_native_chat
        self.ollama_disable_native_chat = ollama_disable_native_chat


class CliMergeTests(unittest.TestCase):
    def test_load_runtime_config_supports_relative_base_config_inheritance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "base.yaml")
            child_path = os.path.join(tmpdir, "child.yaml")
            with open(base_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "OPENAI_API_TYPE": "ollama",
                        "sim_env_id": "highway-fast-v0",
                        "sim_action_target_speeds": [20, 25, 30],
                        "nested": {
                            "keep": 1,
                            "override": 2,
                        },
                    },
                    handle,
                    sort_keys=False,
                )
            with open(child_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "base_config": "base.yaml",
                        "sim_action_target_speeds": [0, 5, 10, 15, 20, 25, 30],
                        "nested": {
                            "override": 3,
                        },
                    },
                    handle,
                    sort_keys=False,
                )

            loaded = load_runtime_config(child_path)

        self.assertEqual(loaded["OPENAI_API_TYPE"], "ollama")
        self.assertEqual(loaded["sim_env_id"], "highway-fast-v0")
        self.assertEqual(loaded["sim_action_target_speeds"], [0, 5, 10, 15, 20, 25, 30])
        self.assertEqual(loaded["nested"]["keep"], 1)
        self.assertEqual(loaded["nested"]["override"], 3)

    def test_measurement_mode_applies_benchmark_ollama_overrides(self):
        config = {
            "OPENAI_API_TYPE": "ollama",
            "OLLAMA_THINK_MODE": "auto",
            "OLLAMA_USE_NATIVE_CHAT": False,
        }
        updated = _apply_measurement_runtime_overrides(
            config,
            _Args(ollama_think_mode="no_think"),
            energy_mode="latency_only",
        )
        self.assertEqual(updated["OLLAMA_THINK_MODE"], "no_think")
        self.assertTrue(updated["OLLAMA_USE_NATIVE_CHAT"])
        self.assertTrue(updated["_benchmark_ollama_runtime_overrides"]["auto_forced_native_chat"])

    def test_standard_eval_mode_leaves_ollama_runtime_config_unchanged(self):
        config = {
            "OPENAI_API_TYPE": "ollama",
            "OLLAMA_THINK_MODE": "auto",
            "OLLAMA_USE_NATIVE_CHAT": False,
        }
        updated = _apply_measurement_runtime_overrides(
            config,
            _Args(ollama_think_mode="no_think", ollama_use_native_chat=True),
            energy_mode="none",
        )
        self.assertEqual(updated["OLLAMA_THINK_MODE"], "auto")
        self.assertFalse(updated["OLLAMA_USE_NATIVE_CHAT"])
        self.assertNotIn("_benchmark_ollama_runtime_overrides", updated)

    def test_benchmark_shim_translates_defaults_to_canonical_eval_cli(self):
        translated = translate_benchmark_args_to_eval_argv(
            ["--config", "config.yaml", "--models", "qwen3:1.7b"]
        )
        self.assertIn("--energy-mode", translated)
        self.assertIn("latency_only", translated)
        self.assertIn("--results-root", translated)
        root_idx = translated.index("--results-root")
        self.assertEqual(translated[root_idx + 1], "results/energy_benchmarks")
        self.assertIn("--experiment-id", translated)
        exp_idx = translated.index("--experiment-id")
        self.assertEqual(translated[exp_idx + 1], "energy_latency_benchmark")

    def test_stop_ablation_config_is_a_compatibility_alias_for_default_stop_capable_profile(self):
        default_cfg = load_runtime_config("config.yaml")
        alias_cfg = load_runtime_config("config.stop_ablation.yaml")

        default_bundle = resolve_simulation_env_bundle(
            default_cfg,
            show_trajectories=False,
            render_agent=False,
        )
        alias_bundle = resolve_simulation_env_bundle(
            alias_cfg,
            show_trajectories=False,
            render_agent=False,
        )

        self.assertEqual(
            list(default_bundle["env_config_snapshot"]["action"]["target_speeds"]),
            list(alias_bundle["env_config_snapshot"]["action"]["target_speeds"]),
        )
        self.assertEqual(default_bundle["env_profile_label"], "default_stop_capable")
        self.assertEqual(alias_bundle["env_profile_label"], "default_stop_capable")

    def test_lightweight_rerun_config_inherits_default_and_relaxes_timeout_ladder(self):
        rerun_cfg = load_runtime_config("config.lightweight_rerun.yaml")

        self.assertEqual(rerun_cfg["OPENAI_API_TYPE"], "ollama")
        self.assertEqual(rerun_cfg["sim_env_id"], "highway-fast-v0")
        self.assertEqual(rerun_cfg["sim_action_target_speeds"], [0, 5, 10, 15, 20, 25, 30])
        self.assertEqual(rerun_cfg["eval_timeout_ladder_sec"], [20, 30, 45])
        self.assertEqual(rerun_cfg["eval_timeout_early_stop_min_decisions"], 8)
        self.assertEqual(rerun_cfg["eval_timeout_early_stop_consecutive_timeout_fallbacks"], 4)
        self.assertEqual(rerun_cfg["eval_timeout_model_quarantine_after_collapses"], 4)


if __name__ == "__main__":
    unittest.main()
