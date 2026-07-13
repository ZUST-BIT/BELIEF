"""Offline tests for maintained cost benchmarking utilities."""

import unittest
from unittest.mock import patch

from test_script.tools import analyze_medrag_costs as cost_analyzer
from test_script.tools import benchmark_pubmedqa_costs as cost_benchmark


class CostBenchmarkTests(unittest.TestCase):
    def test_method_aliases_resolve_to_the_two_maintained_baselines(self):
        self.assertEqual(
            cost_benchmark.normalize_method_names("medrag,i-medrag"),
            ["medrag_pubmed", "imedrag_pubmed"],
        )
        self.assertEqual(
            cost_benchmark.normalize_method_names("all"),
            ["medrag_pubmed", "imedrag_pubmed"],
        )

    def test_experimental_method_name_is_rejected(self):
        with self.assertRaises(ValueError):
            cost_benchmark.normalize_method_names("crag")

    def test_default_dataset_path_exists_in_a_clean_checkout(self):
        benchmark = cost_benchmark.PubMedQACostBenchmark.__new__(
            cost_benchmark.PubMedQACostBenchmark
        )

        resolved = benchmark._resolve_data_path(None)

        self.assertEqual(resolved.replace("\\", "/"), "datasets/pubmedqa.json")

    def test_required_method_initialization_fails_loudly(self):
        benchmark = cost_benchmark.PubMedQACostBenchmark.__new__(
            cost_benchmark.PubMedQACostBenchmark
        )
        benchmark.method_names = ["medrag_pubmed"]

        with patch.object(
            cost_benchmark,
            "MedRAGPubMedRunner",
            side_effect=RuntimeError("backend unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "required method"):
                benchmark._build_methods()

    def test_invalid_predictions_count_as_incorrect(self):
        benchmark = cost_benchmark.PubMedQACostBenchmark.__new__(
            cost_benchmark.PubMedQACostBenchmark
        )
        empty_tokens = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_calls": 0,
            "estimated_calls": 0,
        }
        rows = [
            {
                "prediction": "yes",
                "is_correct": True,
                "elapsed_s": 1.0,
                "token_stats": empty_tokens,
                "extra": {},
                "error": None,
            },
            {
                "prediction": "",
                "is_correct": False,
                "elapsed_s": 1.0,
                "token_stats": empty_tokens,
                "extra": {},
                "error": None,
            },
        ]

        summary = benchmark._summarize_method(rows)

        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["invalid_predictions"], 1)


class CostAnalyzerTests(unittest.TestCase):
    def test_percentile_interpolation_is_deterministic(self):
        self.assertEqual(cost_analyzer.safe_percentile([1.0, 3.0], 0.5), 2.0)
        self.assertIsNone(cost_analyzer.safe_percentile([], 0.95))

    def test_token_estimate_handles_empty_and_non_empty_values(self):
        self.assertEqual(cost_analyzer.estimate_tokens(""), 0)
        self.assertEqual(cost_analyzer.estimate_tokens("abcdefgh"), 2)

    def test_analyzer_counts_invalid_predictions_as_incorrect(self):
        rows = [
            {
                "valid": True,
                "is_correct": True,
                "elapsed_s": None,
                "token_stats": {},
                "retrieval_stats": {},
                "error": None,
            },
            {
                "valid": False,
                "is_correct": False,
                "elapsed_s": None,
                "token_stats": {},
                "retrieval_stats": {},
                "error": None,
            },
        ]

        summary = cost_analyzer.summarize_rows(rows)

        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["invalid_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
