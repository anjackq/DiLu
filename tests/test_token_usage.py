import unittest

from dilu.runtime.token_usage import (
    aggregate_episode_token_usage,
    combine_token_usage_records,
    build_token_usage_record_from_langchain_message,
    build_token_usage_record_from_ollama_native_payload,
)


class _UsageMessage:
    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


class TokenUsageTests(unittest.TestCase):
    def test_native_payload_usage_parses_prompt_and_completion_tokens(self):
        payload = {
            "prompt_eval_count": 17,
            "eval_count": 434,
        }
        usage = build_token_usage_record_from_ollama_native_payload(payload)
        self.assertEqual(
            usage,
            {
                "prompt_tokens": 17,
                "completion_tokens": 434,
                "total_tokens": 451,
                "token_count_method": "ollama_native_usage",
                "token_usage_source": "native_api",
            },
        )

    def test_langchain_message_usage_prefers_usage_metadata(self):
        message = _UsageMessage(
            usage_metadata={
                "input_tokens": 17,
                "output_tokens": 470,
                "total_tokens": 487,
            },
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 99,
                    "completion_tokens": 99,
                    "total_tokens": 198,
                }
            },
        )
        usage = build_token_usage_record_from_langchain_message(message)
        self.assertEqual(
            usage,
            {
                "prompt_tokens": 17,
                "completion_tokens": 470,
                "total_tokens": 487,
                "token_count_method": "ollama_openai_usage",
                "token_usage_source": "openai_compat",
            },
        )

    def test_langchain_message_usage_falls_back_to_response_metadata_token_usage(self):
        message = _UsageMessage(
            usage_metadata=None,
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 34,
                    "total_tokens": 46,
                }
            },
        )
        usage = build_token_usage_record_from_langchain_message(message)
        self.assertEqual(usage["prompt_tokens"], 12)
        self.assertEqual(usage["completion_tokens"], 34)
        self.assertEqual(usage["total_tokens"], 46)
        self.assertEqual(usage["token_count_method"], "ollama_openai_usage")
        self.assertEqual(usage["token_usage_source"], "openai_compat")

    def test_episode_token_aggregation_marks_mixed_when_fallback_is_used(self):
        aggregated = aggregate_episode_token_usage(
            [
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "token_count_method": "ollama_native_usage",
                    "token_usage_source": "native_api",
                },
                {
                    "prompt_tokens": 5,
                    "completion_tokens": 6,
                    "total_tokens": 11,
                    "token_count_method": "whitespace_estimate",
                    "token_usage_source": "estimate_fallback",
                },
            ]
        )
        self.assertEqual(aggregated["prompt_tokens_total"], 15)
        self.assertEqual(aggregated["completion_tokens_total"], 26)
        self.assertEqual(aggregated["total_tokens"], 41)
        self.assertEqual(aggregated["tokens_generated_total"], 26)
        self.assertEqual(aggregated["token_count_method"], "mixed")
        self.assertEqual(aggregated["token_usage_source"], "mixed")

    def test_combine_token_usage_records_sums_primary_and_checker_usage(self):
        combined = combine_token_usage_records(
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "token_count_method": "ollama_native_usage",
                "token_usage_source": "native_api",
            },
            {
                "prompt_tokens": 5,
                "completion_tokens": 6,
                "total_tokens": 11,
                "token_count_method": "ollama_openai_usage",
                "token_usage_source": "openai_compat",
            },
        )
        self.assertEqual(combined["prompt_tokens"], 15)
        self.assertEqual(combined["completion_tokens"], 26)
        self.assertEqual(combined["total_tokens"], 41)
        self.assertEqual(combined["token_count_method"], "mixed")
        self.assertEqual(combined["token_usage_source"], "mixed")

    def test_combine_token_usage_records_ignores_missing_checker_usage(self):
        combined = combine_token_usage_records(
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "token_count_method": "ollama_native_usage",
                "token_usage_source": "native_api",
            },
            None,
        )
        self.assertEqual(combined["prompt_tokens"], 10)
        self.assertEqual(combined["completion_tokens"], 20)
        self.assertEqual(combined["total_tokens"], 30)
        self.assertEqual(combined["token_count_method"], "ollama_native_usage")
        self.assertEqual(combined["token_usage_source"], "native_api")


if __name__ == "__main__":
    unittest.main()
