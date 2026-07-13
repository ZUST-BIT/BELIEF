"""Question-only online PubMed retrieval followed by local BM25 ranking."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pubmed_online import PubMedOnlineSearcher


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "did", "do", "does", "for", "from", "had", "has", "have",
    "how", "if", "in", "into", "is", "it", "may", "most", "not", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were",
    "what", "when", "which", "who", "with", "would",
}


def tokenize_biomedical(text: str) -> List[str]:
    """Tokenize English biomedical text for lexical retrieval."""
    tokens = _TOKEN_RE.findall(str(text or "").lower())
    filtered = [token for token in tokens if token not in _STOP_WORDS]
    return filtered or tokens


def _bm25_scores(
    query_tokens: Sequence[str],
    corpus_tokens: Sequence[Sequence[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """Compute Okapi BM25 scores without requiring a heavyweight search index."""
    document_count = len(corpus_tokens)
    if document_count == 0:
        return []

    lengths = [len(document) for document in corpus_tokens]
    avg_length = sum(lengths) / document_count if document_count else 0.0
    avg_length = avg_length or 1.0
    term_frequencies = [Counter(document) for document in corpus_tokens]

    document_frequency: Counter[str] = Counter()
    for document in corpus_tokens:
        document_frequency.update(set(document))

    scores = [0.0] * document_count
    # Repetition in a natural-language question should not multiply a term's weight.
    for term in dict.fromkeys(query_tokens):
        df = document_frequency.get(term, 0)
        if df == 0:
            continue
        inverse_document_frequency = math.log(
            1.0 + (document_count - df + 0.5) / (df + 0.5)
        )
        for index, frequencies in enumerate(term_frequencies):
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue
            denominator = tf + k1 * (
                1.0 - b + b * lengths[index] / avg_length
            )
            scores[index] += inverse_document_frequency * (
                tf * (k1 + 1.0) / denominator
            )
    return scores


class PubMedBM25Retriever:
    """
    Build an online PubMed candidate pool from the question and rank it with BM25.

    This is intentionally a two-stage online baseline. NCBI ESearch supplies a
    manageable candidate pool; local BM25 then produces the final reproducible
    top-k ordering over title-and-abstract snippets. Neither answer options nor
    gold labels are accepted by this API, preventing retrieval leakage.
    """

    CACHE_VERSION = 1

    def __init__(
        self,
        top_k: int = 5,
        candidate_k: int = 50,
        max_abstract_chars: int = 3000,
        cache_dir: Optional[str | Path] = "cache/medrag_pubmed",
        refresh_cache: bool = False,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        searcher: Optional[Any] = None,
        request_timeout: int = 30,
        retry: int = 2,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if max_abstract_chars <= 0:
            raise ValueError("max_abstract_chars must be greater than zero")

        self.top_k = int(top_k)
        self.candidate_k = int(candidate_k)
        self.max_abstract_chars = int(max_abstract_chars)
        self.refresh_cache = bool(refresh_cache)
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.searcher = searcher or PubMedOnlineSearcher(
            top_k=self.candidate_k,
            max_abstract_len=self.max_abstract_chars,
            request_timeout=request_timeout,
            retry=retry,
        )

    def _cache_path(self, question: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key_payload = {
            "version": self.CACHE_VERSION,
            "question": question,
            "candidate_k": self.candidate_k,
            "max_abstract_chars": self.max_abstract_chars,
        }
        digest = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _load_candidates(self, question: str) -> List[Dict[str, Any]]:
        cache_path = self._cache_path(question)
        if cache_path and cache_path.exists() and not self.refresh_cache:
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                candidates = payload.get("candidates", [])
                if isinstance(candidates, list):
                    return candidates
            except (OSError, ValueError, TypeError):
                pass

        candidates = self.searcher.search(question) or []
        if not isinstance(candidates, list):
            candidates = []

        # Do not persist empty responses: they are often transient network or
        # rate-limit failures and should be retried on the next run.
        if cache_path and candidates:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": self.CACHE_VERSION,
                "question": question,
                "candidate_k": self.candidate_k,
                "candidates": candidates,
            }
            temporary_path = cache_path.with_suffix(".tmp")
            try:
                temporary_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary_path.replace(cache_path)
            except OSError:
                # Caching is an optimization and must not make retrieval fail.
                if temporary_path.exists():
                    temporary_path.unlink(missing_ok=True)
        return candidates

    @staticmethod
    def _normalize_candidates(
        candidates: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        seen_ids = set()
        for offset, paper in enumerate(candidates):
            if not isinstance(paper, dict):
                continue
            document_id = str(
                paper.get("pmid") or paper.get("id") or f"pubmed-candidate-{offset + 1}"
            ).strip()
            if document_id in seen_ids:
                continue
            title = str(paper.get("title") or "Untitled").strip()
            content = str(
                paper.get("abstract") or paper.get("content") or ""
            ).strip()
            if not title and not content:
                continue
            seen_ids.add(document_id)
            normalized.append(
                {"id": document_id, "title": title or "Untitled", "content": content}
            )
        return normalized

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """Return the BM25-ranked PubMed snippets for a question-only query."""
        query = str(question or "").strip()
        if not query:
            raise ValueError("question must not be empty")

        candidates = self._normalize_candidates(self._load_candidates(query))
        if not candidates:
            return []

        corpus_tokens = [
            tokenize_biomedical(f"{paper['title']} {paper['content']}")
            for paper in candidates
        ]
        scores = _bm25_scores(
            tokenize_biomedical(query),
            corpus_tokens,
            k1=self.bm25_k1,
            b=self.bm25_b,
        )
        ranked_indices = sorted(
            range(len(candidates)), key=lambda index: (-scores[index], index)
        )[: self.top_k]

        results: List[Dict[str, Any]] = []
        for rank, index in enumerate(ranked_indices, start=1):
            paper = candidates[index]
            results.append(
                {
                    "rank": rank,
                    "id": paper["id"],
                    "title": paper["title"],
                    "content": paper["content"],
                    "score": float(scores[index]),
                }
            )
        return results


def _truncate(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    shortened = value[:max_chars].rsplit(" ", 1)[0].rstrip()
    return f"{shortened or value[:max_chars].rstrip()} ...[TRUNCATED]"


def format_evidence_context(
    snippets: Sequence[Dict[str, Any]],
    max_chars_per_snippet: int = 1800,
    max_context_chars: int = 10000,
) -> str:
    """Flatten ranked snippets using the MedRAG evidence layout."""
    if max_chars_per_snippet <= 0 or max_context_chars <= 0:
        raise ValueError("evidence context limits must be greater than zero")
    if not snippets:
        return "No relevant PubMed evidence was retrieved."

    parts: List[str] = []
    used_chars = 0
    for fallback_rank, snippet in enumerate(snippets, start=1):
        rank = snippet.get("rank", fallback_rank)
        title = str(snippet.get("title") or "Untitled").strip()
        document_id = str(snippet.get("id") or "Unknown").strip()
        content = _truncate(snippet.get("content", ""), max_chars_per_snippet)
        part = (
            f"[{rank}] Title: {title}\n"
            f"PMID: {document_id}\n"
            f"Evidence: {content or 'No abstract available.'}"
        )
        separator_length = 2 if parts else 0
        if used_chars + separator_length + len(part) > max_context_chars:
            break
        parts.append(part)
        used_chars += separator_length + len(part)

    return "\n\n".join(parts) if parts else "No relevant PubMed evidence was retrieved."
