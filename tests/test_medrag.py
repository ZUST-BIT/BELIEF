import unittest

from medrag import (
    MedRAGPipeline,
    PubMedBM25Retriever,
    format_evidence_context,
    parse_medrag_response,
)
from test_script.medrag_baseline import (
    compute_metrics,
    normalize_example,
)


class FakePubMedSearcher:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [
            {
                "pmid": "1",
                "title": "Insulin treatment in diabetes",
                "abstract": "Insulin lowers blood glucose in diabetes mellitus.",
            },
            {
                "pmid": "2",
                "title": "Cisplatin-associated hearing loss",
                "abstract": "Cisplatin can cause sensorineural hearing loss and tinnitus.",
            },
        ]


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, question):
        self.queries.append(question)
        return [
            {
                "rank": 1,
                "id": "2",
                "title": "Cisplatin ototoxicity",
                "content": "Cisplatin may cause sensorineural hearing loss.",
                "score": 4.2,
            }
        ]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, prompt, temperature=0.7, max_tokens=2000):
        self.calls.append((prompt, temperature, max_tokens))
        return '{"rationale":"The evidence supports cisplatin ototoxicity.","answer":"B"}'


class MedRAGTests(unittest.TestCase):
    def test_question_only_bm25_retrieval(self):
        searcher = FakePubMedSearcher()
        retriever = PubMedBM25Retriever(
            top_k=2, candidate_k=2, cache_dir=None, searcher=searcher
        )
        question = "Can cisplatin cause sensorineural hearing loss?"
        results = retriever.retrieve(question)

        self.assertEqual(searcher.queries, [question])
        self.assertEqual(results[0]["id"], "2")
        self.assertEqual([result["rank"] for result in results], [1, 2])
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_evidence_format_preserves_rank(self):
        context = format_evidence_context(
            [
                {"rank": 1, "id": "11", "title": "First", "content": "Alpha"},
                {"rank": 2, "id": "22", "title": "Second", "content": "Beta"},
            ]
        )
        self.assertLess(context.index("[1] Title: First"), context.index("[2] Title: Second"))
        self.assertIn("PMID: 11\nEvidence: Alpha", context)

    def test_pipeline_uses_question_as_only_retrieval_argument(self):
        retriever = FakeRetriever()
        llm = FakeLLM()
        pipeline = MedRAGPipeline(retriever=retriever, llm=llm)
        output = pipeline.run(
            question="Which drug caused this hearing loss?",
            options={"A": "Metformin", "B": "Cisplatin"},
            sample_id="sample-1",
            dataset_name="medqa",
        )

        self.assertEqual(retriever.queries, ["Which drug caused this hearing loss?"])
        self.assertEqual(output["answer"], "B")
        self.assertTrue(output["valid"])
        prompt, temperature, _ = llm.calls[0]
        self.assertEqual(temperature, 0.0)
        self.assertLess(prompt.index("Retrieved PubMed evidence"), prompt.index("Question:"))
        self.assertIn('"rationale"', prompt)

    def test_response_parser_marks_unparseable_output_invalid(self):
        valid = parse_medrag_response(
            "```json\n{\"rationale\":\"brief\",\"answer\":\"yes\"}\n```",
            ["yes", "no", "maybe"],
        )
        invalid = parse_medrag_response(
            "There is conflicting evidence and no final selection.",
            ["yes", "no", "maybe"],
        )
        self.assertEqual(valid["answer"], "yes")
        self.assertTrue(valid["valid"])
        self.assertEqual(invalid["answer"], "")
        self.assertFalse(invalid["valid"])

    def test_pubmedqa_dataset_context_is_not_part_of_normalized_input(self):
        example = normalize_example(
            {
                "_key": "123",
                "QUESTION": "Does treatment work?",
                "CONTEXTS": ["GOLD ARTICLE TEXT MUST NOT LEAK"],
                "final_decision": "yes",
            },
            "pubmedqa",
        )
        self.assertNotIn("CONTEXTS", example)
        self.assertEqual(example["options"], {"yes": "yes", "no": "no", "maybe": "maybe"})
        self.assertEqual(example["gold"], "yes")

    def test_metrics_count_invalid_as_wrong(self):
        metrics = compute_metrics(
            [
                {"gold": "A", "predicted": "A", "valid": True, "is_correct": True},
                {"gold": "B", "predicted": "", "valid": False, "is_correct": False},
            ],
            ["A", "B"],
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["invalid_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
