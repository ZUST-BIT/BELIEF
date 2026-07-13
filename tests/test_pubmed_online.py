"""Offline tests for PubMed request configuration and input guards."""

import unittest
from unittest.mock import Mock, patch

from pubmed_online import PubMedOnlineSearcher


class PubMedOnlineSearcherTests(unittest.TestCase):
    def test_empty_query_returns_without_an_http_request(self):
        searcher = PubMedOnlineSearcher(top_k=5)

        with patch("pubmed_online.requests.get") as request_get:
            self.assertEqual(searcher.search("   "), [])

        request_get.assert_not_called()

    def test_search_override_controls_candidate_count(self):
        searcher = PubMedOnlineSearcher(top_k=5)
        searcher._esearch = Mock(return_value=["123"])
        searcher._efetch = Mock(return_value=[{"pmid": "123"}])

        result = searcher.search("asthma treatment", top_k=2)

        self.assertEqual(result, [{"pmid": "123"}])
        searcher._esearch.assert_called_once_with("asthma treatment", retmax=2)

    def test_ncbi_identity_and_api_key_are_sent(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"esearchresult": {"idlist": ["123"]}}
        searcher = PubMedOnlineSearcher(
            top_k=1,
            retry=0,
            tool="MEDAR-QA-Test",
            email="maintainer@example.com",
            api_key="test-key",
        )

        with patch("pubmed_online.requests.get", return_value=response) as request_get:
            pmids = searcher._esearch("asthma", retmax=1)

        self.assertEqual(pmids, ["123"])
        request_kwargs = request_get.call_args.kwargs
        self.assertEqual(request_kwargs["params"]["tool"], "MEDAR-QA-Test")
        self.assertEqual(request_kwargs["params"]["email"], "maintainer@example.com")
        self.assertEqual(request_kwargs["params"]["api_key"], "test-key")
        self.assertIn("maintainer@example.com", request_kwargs["headers"]["User-Agent"])


if __name__ == "__main__":
    unittest.main()
