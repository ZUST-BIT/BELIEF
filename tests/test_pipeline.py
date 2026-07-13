"""Offline tests for the single-pass MEDAR-QA pipeline."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import medar_pipeline.pipeline as pipeline_module


class FakeQuestionAnalyzer:
    calls = []

    def run(self, question):
        self.calls.append(question)
        return {
            "frame_of_discernment": ["yes", "no"],
            "extraction": {"elements": {"population": "adults"}},
        }


class FakeEvidenceAnalyzer:
    calls = []

    def run(self, question, evidence):
        self.calls.append((question, list(evidence)))
        return {"analyzed_evidences": []}


class FakeEvidenceEvaluator:
    calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"bpa_list": [], "evaluations": []}


class FakeFusionEngine:
    calls = []

    def run(self, question, fod, bpa_list):
        self.calls.append((question, list(fod), list(bpa_list)))
        return {
            "fusion_result": {
                "fused_bpa": {"uncertainty_theta": 1.0},
                "strategy": "none",
                "conflict_coefficient": 0.0,
            },
            "final_decision": {
                "decision": "UNCERTAIN",
                "confidence": 0.0,
                "reason": "No usable evidence.",
            },
        }


class FakeReportGenerator:
    calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "answer": "maybe",
            "reasoning": "The fused evidence is uncertain.",
            "confidence_score": 0.0,
        }


class FakeDirectReasoningAgent:
    calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "answer": "yes",
            "reasoning": "The direct branch leans positive.",
            "confidence_score": 0.6,
            "directional_tendency": "lean_yes",
        }


class FakeAnswerArbiter:
    calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "final_answer": "yes",
            "reasoning": "The direct branch is more informative.",
            "confidence_score": 0.6,
        }


class SinglePassPipelineTests(unittest.TestCase):
    fake_classes = (
        FakeQuestionAnalyzer,
        FakeEvidenceAnalyzer,
        FakeEvidenceEvaluator,
        FakeFusionEngine,
        FakeReportGenerator,
        FakeDirectReasoningAgent,
        FakeAnswerArbiter,
    )

    def setUp(self):
        for fake_class in self.fake_classes:
            fake_class.calls.clear()
        self.retrieval_calls = []

    def _retrieve(self, question, analysis):
        self.retrieval_calls.append((question, analysis))
        return [{"source": "PubMed", "content": "Retrieved abstract"}]

    def _patch_pipeline(self):
        return patch.multiple(
            pipeline_module,
            QuestionAnalyzer=FakeQuestionAnalyzer,
            EvidenceAnalyzer=FakeEvidenceAnalyzer,
            EvidenceEvaluator=FakeEvidenceEvaluator,
            EvidenceFusionEngine=FakeFusionEngine,
            ReportGenerator=FakeReportGenerator,
            DirectReasoningAgent=FakeDirectReasoningAgent,
            AnswerArbiter=FakeAnswerArbiter,
            retrieve_process=self._retrieve,
        )

    def test_pipeline_runs_each_stage_once_and_writes_one_result(self):
        with tempfile.TemporaryDirectory() as temp_dir, self._patch_pipeline():
            with redirect_stdout(io.StringIO()):
                result = pipeline_module.run_pipeline(
                    question="Does the intervention help?",
                    context="User supplied evidence",
                    task_mode="yes/no",
                    output_dir=temp_dir,
                )

            output_files = list(Path(temp_dir).glob("medar_qa_result_*.json"))
            self.assertEqual(len(output_files), 1)
            with output_files[0].open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), result)

        self.assertEqual(result["task_mode"], "YES_NO")
        self.assertEqual(len(self.retrieval_calls), 1)
        self.assertEqual(len(FakeQuestionAnalyzer.calls), 1)
        self.assertEqual(len(FakeEvidenceAnalyzer.calls), 1)
        self.assertEqual(len(FakeEvidenceEvaluator.calls), 1)
        self.assertEqual(len(FakeFusionEngine.calls), 1)
        self.assertEqual(len(FakeReportGenerator.calls), 1)
        self.assertEqual(len(FakeDirectReasoningAgent.calls), 1)
        self.assertEqual(len(FakeAnswerArbiter.calls), 1)
        self.assertEqual(FakeReportGenerator.calls[0]["task_mode"], "YES_NO")
        self.assertEqual(FakeDirectReasoningAgent.calls[0]["task_mode"], "YES_NO")
        self.assertEqual(FakeAnswerArbiter.calls[0]["task_mode"], "YES_NO")
        self.assertNotIn("total_rounds", result)
        self.assertNotIn("retrieval_history", result)
        self.assertEqual(result["retrieval_summary"]["retrieved_count"], 1)

    def test_invalid_input_fails_before_running_pipeline_stages(self):
        with self._patch_pipeline():
            with self.assertRaises(ValueError):
                pipeline_module.run_pipeline("", task_mode="YES_NO")
            with self.assertRaises(ValueError):
                pipeline_module.run_pipeline("Question?", task_mode="free_text")

        for fake_class in self.fake_classes:
            self.assertEqual(fake_class.calls, [])

    def test_direct_branch_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir, self._patch_pipeline():
            with redirect_stdout(io.StringIO()):
                result = pipeline_module.run_pipeline(
                    question="Does the intervention help?",
                    task_mode="YES_NO",
                    enable_direct_llm_branch=False,
                    output_dir=temp_dir,
                )

        self.assertIsNone(result["direct_llm_result"])
        self.assertIsNone(result["final_aggregated_result"])
        self.assertEqual(FakeDirectReasoningAgent.calls, [])
        self.assertEqual(FakeAnswerArbiter.calls, [])


if __name__ == "__main__":
    unittest.main()
