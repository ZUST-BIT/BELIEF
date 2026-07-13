"""Online PubMed search helper based on NCBI E-utilities.

This module uses the public NCBI Entrez API:
https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests

from config import NCBI_API_KEY, NCBI_EMAIL, NCBI_TOOL


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

class PubMedOnlineSearcher:
    """Search PubMed online with ``esearch`` and fetch article metadata with ``efetch``."""

    _STOP_WORDS = {
        "a",
        "an",
        "the",
        "is",
        "it",
        "its",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "by",
        "but",
        "not",
        "with",
        "from",
        "as",
        "this",
        "that",
        "which",
        "what",
        "who",
        "how",
        "why",
        "when",
        "where",
        "if",
        "whether",
    }
    _BOOLEAN_TOKENS = (" AND ", " OR ", " NOT ")

    def __init__(
        self,
        top_k: int = 5,
        max_abstract_len: int = 1200,
        request_timeout: int = 15,
        retry: int = 2,
        tool: str = NCBI_TOOL,
        email: str = NCBI_EMAIL,
        api_key: str = NCBI_API_KEY,
    ) -> None:
        self.top_k = top_k
        self.max_abstract_len = max_abstract_len
        self.request_timeout = request_timeout
        self.retry = retry
        self.tool = tool
        self.email = email
        self.api_key = api_key

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return PubMed papers for a free-text or PubMed boolean query."""
        retmax = self.top_k if top_k is None else max(0, int(top_k))
        if not query or not query.strip() or retmax == 0:
            return []
        pmids = self._esearch(query, retmax=retmax)
        if not pmids:
            return []
        return self._efetch(pmids)

    def _get(self, url: str, params: Dict[str, Any]) -> Optional[requests.Response]:
        """Send a GET request with retry support."""
        user_agent = f"{self.tool}/1.0"
        if self.email:
            user_agent += f" ({self.email})"
        for attempt in range(self.retry + 1):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers={"User-Agent": user_agent},
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                if attempt < self.retry:
                    time.sleep(1.0)
                    continue

                print(f"[PubMedOnline] request failed for {url}: {exc}")
                return None

        return None

    def _looks_like_boolean_query(self, query: str) -> bool:
        """Return whether the query already contains PubMed boolean syntax."""
        if not query:
            return False

        upper_query = f" {query.upper()} "
        if any(token in upper_query for token in self._BOOLEAN_TOKENS):
            return True

        return bool(re.search(r"\[[^\]]+\]", query))

    @staticmethod
    def _normalize_boolean_query(query: str) -> str:
        return re.sub(r"\s+", " ", query).strip()

    def _clean_query(self, query: str) -> str:
        """Clean a natural-language query while preserving PubMed boolean queries."""
        if self._looks_like_boolean_query(query):
            return self._normalize_boolean_query(query)

        cleaned = re.sub(r"[:\[\]\"?()]", " ", query)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        words = [
            word
            for word in cleaned.split()
            if word.lower() not in self._STOP_WORDS and len(word) > 1
        ]
        return " ".join(words[:12])

    def _esearch(self, query: str, retmax: int) -> List[str]:
        """Call ``esearch`` and return a list of PMIDs."""
        params = {
            "db": "pubmed",
            "term": self._clean_query(query),
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
            "tool": self.tool,
        }
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        response = self._get(ESEARCH_URL, params)
        if response is None:
            return []

        try:
            data = response.json()
        except ValueError as exc:
            print(f"[PubMedOnline] failed to parse esearch JSON: {exc}")
            return []

        return data.get("esearchresult", {}).get("idlist", [])

    def _efetch(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Call ``efetch`` and parse article metadata from the returned XML."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "tool": self.tool,
        }
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        response = self._get(EFETCH_URL, params)
        if response is None:
            return []

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            print(f"[PubMedOnline] failed to parse efetch XML: {exc}")
            return []

        papers = []
        for article in root.iter("PubmedArticle"):
            paper = self._parse_article(article)
            if paper:
                papers.append(paper)
        return papers

    def _parse_article(self, article: ET.Element) -> Optional[Dict[str, Any]]:
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        year_el = article.find(".//PubDate/Year")

        pmid = (
            pmid_el.text.strip() if pmid_el is not None and pmid_el.text else "Unknown"
        )
        title = self._get_text(title_el)
        year = (
            year_el.text.strip() if year_el is not None and year_el.text else "Unknown"
        )

        abstract_parts = []
        for abstract_el in article.findall(".//AbstractText"):
            label = abstract_el.get("Label", "")
            text = self._get_text(abstract_el)
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts).strip()

        if len(abstract) > self.max_abstract_len:
            abstract = abstract[: self.max_abstract_len].rsplit(" ", 1)[0] + "..."

        authors = []
        for author_el in article.findall(".//Author"):
            last_name = author_el.find("LastName")
            if last_name is not None and last_name.text:
                authors.append(last_name.text)

        authors_str = ", ".join(authors[:5])
        if len(authors) > 5:
            authors_str += " et al."

        if not title and not abstract:
            return None

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "year": year,
            "authors": authors_str,
        }

    @staticmethod
    def _get_text(element: Optional[ET.Element]) -> str:
        if element is None:
            return ""
        return "".join(element.itertext()).strip()
