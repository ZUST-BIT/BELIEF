import unittest

from medrag import (
    DEFAULT_I_MEDRAG_PUBMED_CONFIG,
    answer_followup_query_with_pubmed_rag,
    generate_final_answer_from_history,
    run_i_medrag_pubmed,
)


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return [
            {
                "rank": 1,
                "id": "100",
                "title": "Cisplatin ototoxicity",
                "content": "Cisplatin is associated with sensorineural hearing loss.",
                "score": 5.0,
            },
            {
                "rank": 2,
                "id": "101",
                "title": "Platinum DNA adducts",
                "content": "Cisplatin exerts antitumor effects by cross-linking DNA.",
                "score": 4.0,
            },
        ]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, prompt, temperature=0.7, max_tokens=2000):
        self.calls.append((prompt, temperature, max_tokens))
        if "generate follow-up search queries" in prompt:
            return (
                '{"queries": ['
                '"cisplatin sensorineural hearing loss mechanism", '
                '"cisplatin DNA cross-linking bladder cancer"'
                "]}"
            )
        if "Answer the follow-up medical query" in prompt:
            return (
                '{"answer": "Retrieved PubMed evidence supports cisplatin ototoxicity '
                'and DNA cross-linking.", '
                '"evidence_summary": "Cisplatin can cause hearing loss and forms DNA adducts.", '
                '"used_pmids": ["100", "101"]}'
            )
        if "choose the single best answer" in prompt:
            return (
                '{"answer_choice": "D", '
                '"answer_text": "Cross-linking of DNA", '
                '"rationale": "The history links cisplatin to ototoxicity and DNA cross-linking."}'
            )
        return "{}"


class IMedRAGPubMedTests(unittest.TestCase):
    def test_followup_answer_uses_pubmed_retrieval(self):
        retriever = FakeRetriever()
        llm = FakeLLM()
        result = answer_followup_query_with_pubmed_rag(
            query="cisplatin ototoxicity",
            pubmed_retriever=retriever,
            llm=llm,
            k=2,
            temperature=0.2,
            max_tokens=128,
            llm_cache=None,
        )

        self.assertEqual(retriever.queries, ["cisplatin ototoxicity"])
        self.assertEqual(result["retrieved_pmids"], ["100", "101"])
        self.assertIn("cisplatin", result["answer"].lower())
        self.assertEqual(result["used_pmids"], ["100", "101"])

    def test_final_answer_is_normalized_to_valid_choice(self):
        llm = FakeLLM()
        result = generate_final_answer_from_history(
            question="Which action explains cisplatin benefit?",
            options={
                "A": "Inhibition of proteasome",
                "B": "Hyperstabilization of microtubules",
                "C": "Generation of free radicals",
                "D": "Cross-linking of DNA",
            },
            qa_history=[
                {
                    "query": "cisplatin DNA cross-linking",
                    "answer": "Cisplatin cross-links DNA.",
                    "pmids": ["101"],
                }
            ],
            dataset_name="medqa",
            llm=llm,
            temperature=0.2,
            max_tokens=128,
            max_history_items=8,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["answer_choice"], "D")
        self.assertEqual(result["answer_text"], "Cross-linking of DNA")

    def test_run_i_medrag_pubmed_toy_flow(self):
        retriever = FakeRetriever()
        llm = FakeLLM()
        config = dict(DEFAULT_I_MEDRAG_PUBMED_CONFIG)
        config.update(
            {
                "n_rounds": 1,
                "n_queries": 2,
                "k_per_query": 2,
                "cache_llm_outputs": False,
                "cache_retrieval": False,
                "max_tokens_query": 128,
                "max_tokens_answer": 128,
                "max_tokens_final": 128,
            }
        )
        result = run_i_medrag_pubmed(
            question=(
                "A patient receives chemotherapy and develops sensorineural "
                "hearing loss. Which drug action explains the benefit?"
            ),
            options={
                "A": "Inhibition of proteasome",
                "B": "Hyperstabilization of microtubules",
                "C": "Generation of free radicals",
                "D": "Cross-linking of DNA",
            },
            dataset_name="medqa",
            llm=llm,
            pubmed_retriever=retriever,
            config=config,
        )

        self.assertEqual(result["method"], "i-MedRAG-PubMed")
        self.assertEqual(result["final_prediction"], "D")
        self.assertTrue(result["valid"])
        self.assertEqual(result["num_followup_rounds"], 1)
        self.assertEqual(result["num_followup_queries"], 2)
        self.assertEqual(result["num_pubmed_queries"], 2)
        self.assertEqual(len(result["qa_history"]), 2)
        self.assertEqual(len(retriever.queries), 2)
        self.assertGreater(result["input_tokens"], 0)
        self.assertGreater(result["output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()

