"""Offline regression tests for shared MEDAR-QA contracts."""

import unittest

import agents
import medar_agents
from medar_agents.evidence_fusion_engine import EvidenceFusionEngine
from medar_agents.json_utils import extract_json_from_response
from medar_agents.task_modes import normalize_task_mode


class JsonParsingContractTests(unittest.TestCase):
    def test_parser_accepts_an_embedded_json_object(self):
        result = extract_json_from_response('prefix {"answer": "yes"} suffix')

        self.assertEqual(result, {"answer": "yes"})

    def test_parser_rejects_non_object_json_values(self):
        for response in ('[1, 2]', '"answer"', 'null', '42', ''):
            with self.subTest(response=response):
                self.assertIsNone(extract_json_from_response(response))


class FusionContractTests(unittest.TestCase):
    def test_empty_evidence_is_explicitly_uncertain(self):
        result = EvidenceFusionEngine().run(
            question="Does the evidence support the claim?",
            fod=["yes", "no"],
            bpa_list=[],
        )

        self.assertEqual(
            result["fusion_result"]["fused_bpa"],
            {"uncertainty_theta": 1.0},
        )
        self.assertEqual(result["fusion_result"]["strategy"], "none")
        self.assertEqual(result["final_decision"]["decision"], "UNCERTAIN")
        self.assertEqual(result["final_decision"]["confidence"], 0.0)


class PublicApiContractTests(unittest.TestCase):
    def test_compatibility_module_exposes_the_canonical_public_api(self):
        self.assertEqual(set(agents.__all__), set(medar_agents.__all__))
        for name in medar_agents.__all__:
            self.assertTrue(hasattr(agents, name), name)

    def test_task_mode_aliases_are_normalized_and_invalid_modes_fail(self):
        self.assertEqual(normalize_task_mode("yes/no"), "YES_NO")
        self.assertEqual(normalize_task_mode("yes-no"), "YES_NO")
        self.assertEqual(normalize_task_mode("selection"), "SELECTION")
        with self.assertRaises(ValueError):
            normalize_task_mode("free_text")


if __name__ == "__main__":
    unittest.main()
