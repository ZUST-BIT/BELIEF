"""Tests for the unified baseline command router."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import run_experiment


class RunExperimentRouterTests(unittest.TestCase):
    def test_generic_help_lists_both_supported_methods(self):
        output = io.StringIO()

        with redirect_stdout(output):
            run_experiment.main(["--help"])

        help_text = output.getvalue()
        self.assertIn("MedRAG-PubMed", help_text)
        self.assertIn("i-MedRAG-PubMed", help_text)
        self.assertIn("method and add --help", help_text)

    def test_medrag_alias_routes_to_medrag_runner(self):
        with patch.object(run_experiment, "_run_medrag") as run_medrag:
            run_experiment.main([
                "--method",
                "medrag-pubmed",
                "--dataset",
                "PubMedQA",
                "--limit",
                "1",
            ])

        run_medrag.assert_called_once()
        routed_args = run_medrag.call_args.args[0]
        self.assertNotIn("--method", routed_args)
        self.assertIn("--dataset", routed_args)
        self.assertIn("pubmedqa", routed_args)

    def test_imedrag_alias_routes_to_imedrag_runner(self):
        with patch.object(run_experiment, "_run_imedrag") as run_imedrag:
            run_experiment.main([
                "--method=i-medrag-pubmed",
                "--dataset=MedQA",
                "--limit=1",
            ])

        run_imedrag.assert_called_once()
        routed_args = run_imedrag.call_args.args[0]
        self.assertIn("--method=i-MedRAG-PubMed", routed_args)
        self.assertIn("--dataset=medqa", routed_args)


if __name__ == "__main__":
    unittest.main()
