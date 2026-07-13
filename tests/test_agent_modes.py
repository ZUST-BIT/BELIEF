"""Offline tests for task-mode routing and stable Agent output contracts."""

import json
import unittest

from medar_agents.answer_arbiter import AnswerArbiter
from medar_agents.direct_reasoning_agent import DirectReasoningAgent
from medar_agents.report_generator import ReportGenerator


class FakeChain:
    """Minimal chain replacement that records prompts and returns fixed responses."""

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("FakeChain has no response left")
        return self.responses.pop(0)


class DirectReasoningAgentModeTests(unittest.TestCase):
    def test_invalid_task_mode_is_rejected(self):
        agent = DirectReasoningAgent()

        with self.assertRaises(ValueError):
            agent.run("Question?", {"analyzed_evidences": []}, task_mode="free_text")

    def test_default_mode_uses_mcq_prompt(self):
        agent = DirectReasoningAgent()
        chain = FakeChain(json.dumps({
            "selected_option": "b",
            "reasoning": "Option B best fits.",
            "confidence_score": 0.8,
        }))
        agent._chain = chain

        result = agent.run("Which option is correct?", {"analyzed_evidences": []})

        self.assertEqual(result["selected_option"], "B")
        self.assertIn("Option Elimination", chain.prompts[0])
        self.assertNotIn("EVIDENCE RELATIONSHIP ANALYSIS", chain.prompts[0])

    def test_yesno_mode_uses_yesno_prompt_and_preserves_strong_tendency(self):
        agent = DirectReasoningAgent()
        chain = FakeChain(json.dumps({
            "answer": "yes",
            "reasoning": "The evidence converges.",
            "confidence_score": 0.82,
            "directional_tendency": "strong_yes",
        }))
        agent._chain = chain

        result = agent.run(
            "Does the intervention help?",
            {"analyzed_evidences": []},
            task_mode="YES_NO",
        )

        self.assertEqual(result["answer"], "yes")
        self.assertEqual(result["directional_tendency"], "strong_yes")
        self.assertIn("EVIDENCE RELATIONSHIP ANALYSIS", chain.prompts[0])

    def test_invalid_response_uses_mode_specific_fallback(self):
        agent = DirectReasoningAgent()
        agent._chain = FakeChain("not valid json")

        result = agent.run(
            "Does the intervention help?",
            {"analyzed_evidences": []},
            task_mode="YES_NO",
        )

        self.assertEqual(result["answer"], "maybe")
        self.assertEqual(result["confidence_score"], 0.0)
        self.assertIn("error", result)

    def test_non_finite_confidence_is_rejected(self):
        agent = DirectReasoningAgent()
        agent._chain = FakeChain(json.dumps({
            "answer": "yes",
            "confidence_score": "nan",
        }))

        result = agent.run(
            "Does the intervention help?",
            {"analyzed_evidences": []},
            task_mode="YES_NO",
        )

        self.assertEqual(result["confidence_score"], 0.0)


class ReportGeneratorModeTests(unittest.TestCase):
    def test_default_mode_uses_mcq_prompt_and_limits_evidence_size(self):
        agent = ReportGenerator()
        chain = FakeChain(json.dumps({
            "answer": "A",
            "reasoning": "The fused evidence favors A.",
            "confidence_score": 0.7,
        }))
        agent._chain = chain
        long_evidence = "x" * 4500

        result = agent.run(
            "Which option is correct?",
            {"decision": "A", "confidence": 0.7},
            {},
            [{"content": long_evidence}],
        )

        self.assertEqual(result["answer"], "A")
        self.assertIn("Always Produce a Final Choice", chain.prompts[0])
        self.assertIn("x" * 4000, chain.prompts[0])
        self.assertNotIn("x" * 4001, chain.prompts[0])
        self.assertNotIn("{{REASONING_HISTORY}}", chain.prompts[0])

    def test_enhanced_evidence_source_is_preserved(self):
        agent = ReportGenerator()
        chain = FakeChain(json.dumps({
            "answer": "A",
            "reasoning": "The evidence supports A.",
            "confidence_score": 0.7,
        }))
        agent._chain = chain

        agent.run(
            "Which option is correct?",
            {"decision": "A", "confidence": 0.7},
            {},
            [{"source": "PubMed", "content": "Evidence"}],
        )

        self.assertIn('"source_type": "PubMed"', chain.prompts[0])

    def test_yesno_mode_uses_yesno_prompt_and_stable_contract(self):
        agent = ReportGenerator()
        chain = FakeChain(json.dumps({
            "answer": "strong_no",
            "reasoning": "The fused direction is negative.",
            "confidence_score": "0.75",
        }))
        agent._chain = chain

        result = agent.run(
            "Does the intervention help?",
            {"decision": "no"},
            {},
            [],
            task_mode="YES_NO",
        )

        self.assertEqual(result["answer"], "no")
        self.assertEqual(result["reasoning"], "The fused direction is negative.")
        self.assertEqual(result["confidence_score"], 0.75)
        self.assertIn("Quantify Doubt", chain.prompts[0])

    def test_parse_failure_keeps_the_same_output_contract(self):
        agent = ReportGenerator()
        agent._chain = FakeChain("plain-text fallback reasoning")

        result = agent.run(
            "Does the intervention help?",
            {"answer": "strong_yes", "confidence": 1.5},
            {},
            [],
            task_mode="YES_NO",
        )

        self.assertEqual(result["answer"], "yes")
        self.assertEqual(result["reasoning"], "plain-text fallback reasoning")
        self.assertEqual(result["confidence_score"], 1.0)
        self.assertTrue({"answer", "reasoning", "confidence_score"}.issubset(result))
        self.assertIn("error", result)

    def test_parse_failure_maps_real_ds_yesno_label(self):
        agent = ReportGenerator()
        agent._chain = FakeChain("plain-text fallback reasoning")

        result = agent.run(
            "Does the intervention help?",
            {"decision": "SUPPORT_ASSOCIATION", "confidence": 0.82},
            {},
            [],
            task_mode="YES_NO",
        )

        self.assertEqual(result["answer"], "yes")
        self.assertEqual(result["confidence_score"], 0.82)
        self.assertEqual(
            agent._normalize_answer("FAVOR_INTERVENTION", is_mcq=False),
            "yes",
        )
        self.assertEqual(
            agent._normalize_answer("FAVOR_COMPARATOR", is_mcq=False),
            "no",
        )
        self.assertEqual(
            agent._normalize_answer("NO_SIGNIFICANT_DIFFERENCE", is_mcq=False),
            "no",
        )

    def test_parse_failure_maps_mcq_option_text_to_label(self):
        agent = ReportGenerator()
        agent._chain = FakeChain("plain-text fallback reasoning")
        question = '''Which condition is most likely?
        "A": "Renal cell carcinoma"
        "B": "Meningioma"
        "C": "Astrocytoma"
        "D": "Vascular malformation"'''

        result = agent.run(
            question,
            {"decision": "Meningioma", "confidence": 0.78},
            {},
            [],
        )

        self.assertEqual(result["answer"], "B")
        self.assertEqual(result["confidence_score"], 0.78)

    def test_mcq_option_text_mapping_accepts_parenthesized_options(self):
        options = ReportGenerator._extract_mcq_options(
            "Which condition?\n(A) Astrocytoma\n(B) Meningioma"
        )

        self.assertEqual(options, {"A": "Astrocytoma", "B": "Meningioma"})

    def test_non_finite_report_confidence_is_rejected(self):
        agent = ReportGenerator()
        agent._chain = FakeChain(json.dumps({
            "answer": "yes",
            "reasoning": "Positive direction.",
            "confidence_score": "Infinity",
        }))

        result = agent.run(
            "Does the intervention help?",
            {"decision": "SUPPORT_ASSOCIATION"},
            {},
            [],
            task_mode="YES_NO",
        )

        self.assertEqual(result["confidence_score"], 0.0)


class AnswerArbiterModeTests(unittest.TestCase):
    def test_default_mode_uses_mcq_prompt(self):
        agent = AnswerArbiter()
        chain = FakeChain(json.dumps({
            "final_answer": "B",
            "reasoning": "Both branches support B.",
            "confidence_score": 0.8,
        }))
        agent._chain = chain

        result = agent.run(
            "Which option is correct?",
            {"selected_option": "B", "confidence_score": 0.8},
            {"selected_option": "B", "confidence_score": 0.8},
        )

        self.assertEqual(result["final_answer"], "B")
        self.assertIn("Chief Medical Arbitration Panel", chain.prompts[0])
        self.assertNotIn("Independent Critical Review", chain.prompts[0])

    def test_yesno_mode_passes_mode_to_prompt_builder_and_normalizes_strong_values(self):
        agent = AnswerArbiter()
        chain = FakeChain(json.dumps({
            "final_answer": "strong_no",
            "reasoning": "The negative branch is stronger.",
            "confidence_score": 0.7,
        }))
        agent._chain = chain

        result = agent.run(
            "Does the intervention help?",
            {"answer": "strong_no", "confidence_score": 0.7},
            {"answer": "no", "confidence_score": 0.7},
            task_mode="YES_NO",
        )

        self.assertEqual(result["final_answer"], "no")
        self.assertIn("Independent Critical Review", chain.prompts[0])
        self.assertEqual(agent._normalize_yesno_answer("strong_yes"), "yes")
        self.assertEqual(agent._normalize_yesno_answer("strong_no"), "no")
        self.assertEqual(agent._normalize_yesno_answer("SUPPORT_ASSOCIATION"), "yes")
        self.assertEqual(agent._normalize_yesno_answer("REFUTE_ASSOCIATION"), "no")
        self.assertEqual(agent._normalize_yesno_answer("FAVOR_INTERVENTION"), "yes")
        self.assertEqual(agent._normalize_yesno_answer("FAVOR_COMPARATOR"), "no")
        self.assertEqual(agent._map_answer_to_score("maybe", "strong_yes"), 0.6)
        self.assertEqual(agent._map_answer_to_score("maybe", "strong_no"), -0.6)

    def test_yesno_parse_failure_uses_arbitration_recommendation(self):
        agent = AnswerArbiter()
        chain = FakeChain("not valid json")
        agent._chain = chain

        result = agent.run(
            "Does the intervention help?",
            {"answer": "maybe", "confidence_score": 0.0},
            {
                "answer": "maybe",
                "confidence_score": 1.0,
                "directional_tendency": "strong_yes",
            },
            task_mode="YES_NO",
        )

        self.assertEqual(result["final_answer"], "yes")
        self.assertEqual(result["arbitration_recommendation"], "yes")
        self.assertEqual(result["recommended_source"], "LLM")
        self.assertIn('"recommended_answer": "yes"', chain.prompts[0])

    def test_negative_and_non_finite_confidences_do_not_break_arbitration(self):
        agent = AnswerArbiter()
        agent._chain = FakeChain(json.dumps({
            "final_answer": "maybe",
            "confidence_score": "nan",
        }))

        result = agent.run(
            "Does the intervention help?",
            {"answer": "yes", "confidence_score": -1.0},
            {"answer": "no", "confidence_score": "-Infinity"},
            task_mode="YES_NO",
        )

        self.assertEqual(result["final_answer"], "maybe")
        self.assertEqual(result["confidence_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
