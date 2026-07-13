"""i-MedRAG-PubMed baseline.

This module implements iterative follow-up query generation on top of the
existing PubMed retriever and LLM client. It deliberately avoids the BELIEF
structured evidence, uncertainty fusion, and arbitration components.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from llm_client import BaseLLMClient, get_llm_client

from .retrieval import PubMedBM25Retriever


METHOD_NAME = "i-MedRAG-PubMed"

DEFAULT_I_MEDRAG_PUBMED_CONFIG: Dict[str, Any] = {
    "n_rounds": 2,
    "n_queries": 2,
    "k_per_query": 5,
    "max_history_items": 8,
    "temperature_query": 0.2,
    "temperature_answer": 0.2,
    "temperature_final": 0.2,
    "max_tokens_query": 512,
    "max_tokens_answer": 512,
    "max_tokens_final": 512,
    "deduplicate_pmids": True,
    "cache_retrieval": True,
    "cache_llm_outputs": True,
    "candidate_k": 50,
    "max_abstract_chars": 3000,
    "max_chars_per_snippet": 1200,
    "max_context_chars": 8000,
    "retrieval_cache_dir": "cache/medrag_pubmed",
    "llm_cache_dir": "cache/imedrag_pubmed_llm",
}

_CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "may",
    "most",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "with",
    "would",
}


def _merge_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_I_MEDRAG_PUBMED_CONFIG)
    if config:
        merged.update(dict(config))

    for key in (
        "n_rounds",
        "n_queries",
        "k_per_query",
        "max_history_items",
        "max_tokens_query",
        "max_tokens_answer",
        "max_tokens_final",
        "candidate_k",
        "max_abstract_chars",
        "max_chars_per_snippet",
        "max_context_chars",
    ):
        merged[key] = int(merged[key])
    for key in ("temperature_query", "temperature_answer", "temperature_final"):
        merged[key] = float(merged[key])
    for key in ("deduplicate_pmids", "cache_retrieval", "cache_llm_outputs"):
        merged[key] = bool(merged[key])

    if merged["n_rounds"] <= 0:
        raise ValueError("n_rounds must be greater than zero")
    if merged["n_queries"] <= 0:
        raise ValueError("n_queries must be greater than zero")
    if merged["k_per_query"] <= 0:
        raise ValueError("k_per_query must be greater than zero")
    return merged


def _normalize_options(
    options: Optional[Mapping[str, Any] | Sequence[Any]],
    dataset_name: str,
) -> Dict[str, str]:
    dataset_key = str(dataset_name or "").strip().lower()
    if dataset_key == "pubmedqa":
        return {"yes": "yes", "no": "no", "maybe": "maybe"}

    if isinstance(options, Mapping):
        normalized = {
            str(label).strip(): str(text).strip()
            for label, text in options.items()
            if str(label).strip()
        }
        if normalized:
            return normalized

    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        normalized = {}
        for index, value in enumerate(options):
            if index >= len(_CHOICE_LABELS):
                break
            normalized[_CHOICE_LABELS[index]] = str(value).strip()
        if normalized:
            return normalized

    raise ValueError("multiple-choice examples must provide candidate options")


def _format_options(options: Mapping[str, str]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in options.items())


def _truncate(text: Any, max_chars: int = 1200) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    shortened = value[:max_chars].rsplit(" ", 1)[0].rstrip()
    return f"{shortened or value[:max_chars].rstrip()} ...[TRUNCATED]"


def _estimate_tokens(text: Any) -> int:
    value = str(text or "")
    if not value:
        return 0
    return max(1, math.ceil(len(value) / 4))


def _empty_usage() -> Dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated": True,
        "by_stage": {},
    }


def _add_usage(
    total: Dict[str, Any],
    stage: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    total["input_tokens"] += int(input_tokens)
    total["output_tokens"] += int(output_tokens)
    stage_usage = total["by_stage"].setdefault(
        stage, {"input_tokens": 0, "output_tokens": 0}
    )
    stage_usage["input_tokens"] += int(input_tokens)
    stage_usage["output_tokens"] += int(output_tokens)


def _strip_code_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _json_objects(text: str) -> List[str]:
    objects: List[str] = []
    start: Optional[int] = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(str(text or "")):
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_string:
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return sorted(objects, key=len, reverse=True)


def _first_json_object(raw_response: Any) -> Optional[Dict[str, Any]]:
    raw = str(raw_response or "").strip()
    candidates = [_strip_code_fences(raw)] + [
        _strip_code_fences(candidate) for candidate in _json_objects(raw)
    ]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _fallback_pubmed_query(question: str) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]+", str(question or ""))
    filtered = [
        token
        for token in tokens
        if token.lower() not in _QUERY_STOP_WORDS and len(token) > 1
    ]
    query = " ".join(filtered[:16]).strip()
    return query or str(question or "").strip() or "biomedical evidence"


def _normalize_query(query: Any) -> str:
    return re.sub(r"\s+", " ", str(query or "").strip()).lower()


def _format_history(
    qa_history: Sequence[Mapping[str, Any]],
    max_items: Optional[int] = None,
) -> str:
    if not qa_history:
        return "None."
    selected = list(qa_history)
    if max_items and max_items > 0:
        selected = selected[-max_items:]

    parts = []
    for index, item in enumerate(selected, start=1):
        pmids = item.get("pmids") or item.get("used_pmids") or []
        if isinstance(pmids, (str, bytes)):
            pmid_text = str(pmids)
        else:
            pmid_text = ", ".join(str(pmid) for pmid in pmids)
        evidence_summary = str(item.get("evidence_summary") or "").strip()
        summary_line = (
            f"\n   Evidence summary: {_truncate(evidence_summary, 500)}"
            if evidence_summary
            else ""
        )
        parts.append(
            f"{index}. Query: {_truncate(item.get('query', ''), 400)}\n"
            f"   Answer: {_truncate(item.get('answer', ''), 700)}"
            f"{summary_line}\n"
            f"   Evidence PMIDs: {pmid_text or 'None'}"
        )
    return "\n\n".join(parts)


def build_followup_query_prompt(
    question: str,
    options: Mapping[str, str],
    qa_history: Sequence[Mapping[str, Any]],
    n_queries: int,
    max_history_items: Optional[int] = None,
) -> str:
    return f"""You are helping a medical retrieval-augmented QA system answer a closed-set biomedical question.

Your task is NOT to answer the original question directly.
Your task is to generate follow-up search queries that would help retrieve useful biomedical evidence from PubMed.

Original question:
{question}

Candidate answers:
{_format_options(options)}

Previous information-seeking history:
{_format_history(qa_history, max_items=max_history_items)}

Generate {n_queries} new standalone PubMed search queries.

Requirements:
1. Each query must be medically specific and searchable in PubMed.
2. Each query must target missing information needed to distinguish the candidate answers.
3. Do not repeat previous queries.
4. Do not include answer labels such as A/B/C/D unless necessary as biomedical terms.
5. Do not reveal or assume the ground-truth answer.
6. Return JSON only:
{{
  "queries": [
    "...",
    "..."
  ]
}}"""


def _parse_followup_queries_response(
    raw_response: Any,
    n_queries: int,
) -> List[str]:
    parsed = _first_json_object(raw_response)
    queries: List[str] = []
    if parsed is not None:
        raw_queries = parsed.get("queries", [])
        if isinstance(raw_queries, str):
            raw_queries = [raw_queries]
        if isinstance(raw_queries, Iterable):
            queries.extend(str(item).strip() for item in raw_queries)

    if not queries:
        array_match = re.search(r"\[[\s\S]*\]", str(raw_response or ""))
        if array_match:
            try:
                value = json.loads(array_match.group(0))
            except (TypeError, json.JSONDecodeError):
                value = None
            if isinstance(value, list):
                queries.extend(str(item).strip() for item in value)

    if not queries:
        for line in str(raw_response or "").splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            cleaned = cleaned.strip("\"',")
            if cleaned and len(cleaned.split()) >= 2:
                queries.append(cleaned)

    unique: List[str] = []
    seen = set()
    for query in queries:
        normalized = _normalize_query(query)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(query.strip())
        if len(unique) >= n_queries:
            break
    return unique


class _LLMCallCache:
    def __init__(self, enabled: bool, cache_dir: str | Path) -> None:
        self.enabled = bool(enabled)
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def _llm_id(llm: BaseLLMClient) -> str:
        model = getattr(llm, "model", None)
        if model:
            return f"{llm.__class__.__name__}:{model}"
        return llm.__class__.__name__

    def _cache_path(
        self,
        llm: BaseLLMClient,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
    ) -> Path:
        payload = {
            "stage": stage,
            "llm": self._llm_id(llm),
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def chat(
        self,
        llm: BaseLLMClient,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        cache_path = self._cache_path(llm, prompt, temperature, max_tokens, stage)
        if self.enabled and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                raw = str(payload.get("raw_response", ""))
                return {
                    "raw_response": raw,
                    "cached": True,
                    "latency_seconds": time.perf_counter() - started,
                    "input_tokens": _estimate_tokens(prompt),
                    "output_tokens": _estimate_tokens(raw),
                }
            except (OSError, TypeError, ValueError):
                pass

        raw = llm.chat(prompt, temperature=temperature, max_tokens=max_tokens)
        latency = time.perf_counter() - started
        if self.enabled:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = cache_path.with_suffix(".tmp")
                temporary_path.write_text(
                    json.dumps(
                        {
                            "raw_response": raw,
                            "stage": stage,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "llm": self._llm_id(llm),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                temporary_path.replace(cache_path)
            except OSError:
                pass
        return {
            "raw_response": raw,
            "cached": False,
            "latency_seconds": latency,
            "input_tokens": _estimate_tokens(prompt),
            "output_tokens": _estimate_tokens(raw),
        }


def generate_followup_queries(
    question: str,
    options: Mapping[str, Any] | Sequence[Any],
    qa_history: List[Mapping[str, Any]],
    llm: BaseLLMClient,
    n_queries: int,
    temperature: float,
    max_tokens: int = 512,
) -> List[str]:
    """Generate standalone PubMed follow-up search queries."""
    normalized_options = _normalize_options(options, dataset_name="")
    prompt = build_followup_query_prompt(
        question=question,
        options=normalized_options,
        qa_history=qa_history,
        n_queries=n_queries,
    )
    raw_response = llm.chat(prompt, temperature=temperature, max_tokens=max_tokens)
    queries = _parse_followup_queries_response(raw_response, n_queries=n_queries)
    return queries or [_fallback_pubmed_query(question)]


def _generate_followup_queries_with_metadata(
    question: str,
    options: Mapping[str, str],
    qa_history: Sequence[Mapping[str, Any]],
    llm: BaseLLMClient,
    n_queries: int,
    temperature: float,
    max_tokens: int,
    max_history_items: int,
    llm_cache: _LLMCallCache,
) -> Dict[str, Any]:
    prompt = build_followup_query_prompt(
        question=question,
        options=options,
        qa_history=qa_history,
        n_queries=n_queries,
        max_history_items=max_history_items,
    )
    call = llm_cache.chat(
        llm=llm,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stage="followup_query_generation",
    )
    raw_response = call["raw_response"]
    queries = _parse_followup_queries_response(raw_response, n_queries=n_queries)
    if not queries:
        queries = [_fallback_pubmed_query(question)]
    return {
        "queries": queries,
        "prompt": prompt,
        "raw_model_output": raw_response,
        "cached": call["cached"],
        "latency_seconds": call["latency_seconds"],
        "input_tokens": call["input_tokens"],
        "output_tokens": call["output_tokens"],
    }


def _retrieve_with_k(
    pubmed_retriever: Any,
    query: str,
    k: int,
) -> List[Dict[str, Any]]:
    retrieve = getattr(pubmed_retriever, "retrieve", None)
    if retrieve is None:
        raise ValueError("pubmed_retriever must expose a retrieve(query) method")

    try:
        signature = inspect.signature(retrieve)
        parameters = signature.parameters
    except (TypeError, ValueError):
        parameters = {}

    if "top_k" in parameters:
        return list(retrieve(query, top_k=k) or [])
    if "k" in parameters:
        return list(retrieve(query, k=k) or [])

    old_top_k = getattr(pubmed_retriever, "top_k", None)
    changed_top_k = old_top_k is not None
    if changed_top_k:
        pubmed_retriever.top_k = int(k)
    try:
        return list(retrieve(query) or [])
    finally:
        if changed_top_k:
            pubmed_retriever.top_k = old_top_k


def _deduplicate_retrieval(
    retrieval: Sequence[Mapping[str, Any]],
    deduplicate_pmids: bool = True,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen = set()
    for fallback_rank, item in enumerate(retrieval, start=1):
        if not isinstance(item, Mapping):
            continue
        document_id = str(
            item.get("id") or item.get("pmid") or f"pubmed-{fallback_rank}"
        ).strip()
        if deduplicate_pmids and document_id in seen:
            continue
        seen.add(document_id)
        results.append(
            {
                **dict(item),
                "rank": len(results) + 1,
                "id": document_id,
                "title": str(item.get("title") or "Untitled").strip(),
                "content": str(
                    item.get("content") or item.get("abstract") or item.get("snippet") or ""
                ).strip(),
            }
        )
    return results


def _format_pubmed_evidence(
    retrieval: Sequence[Mapping[str, Any]],
    max_chars_per_snippet: int,
    max_context_chars: int,
) -> str:
    if not retrieval:
        return "No relevant PubMed evidence was retrieved."

    parts: List[str] = []
    used_chars = 0
    for fallback_rank, item in enumerate(retrieval, start=1):
        rank = item.get("rank", fallback_rank)
        document_id = str(item.get("id") or item.get("pmid") or "Unknown").strip()
        title = str(item.get("title") or "Untitled").strip()
        content = _truncate(
            item.get("content") or item.get("abstract") or item.get("snippet") or "",
            max_chars=max_chars_per_snippet,
        )
        part = (
            f"[{rank}] PMID: {document_id}\n"
            f"Title: {title}\n"
            f"Abstract/Snippet: {content or 'No abstract available.'}"
        )
        separator_length = 2 if parts else 0
        if used_chars + separator_length + len(part) > max_context_chars:
            break
        parts.append(part)
        used_chars += separator_length + len(part)
    return "\n\n".join(parts) if parts else "No relevant PubMed evidence was retrieved."


def build_followup_answer_prompt(query: str, evidence_context: str) -> str:
    return f"""You are a biomedical evidence assistant.

Answer the follow-up medical query using only the retrieved PubMed evidence below.
If the evidence is insufficient, say that the retrieved evidence is insufficient.

Follow-up query:
{query}

Retrieved PubMed evidence:
{evidence_context}

Return JSON only:
{{
  "answer": "A concise evidence-grounded answer to the follow-up query.",
  "evidence_summary": "Brief summary of the most relevant retrieved evidence.",
  "used_pmids": ["...", "..."]
}}"""


def _parse_followup_answer_response(
    raw_response: Any,
    retrieved_pmids: Sequence[str],
) -> Dict[str, Any]:
    parsed = _first_json_object(raw_response)
    retrieved_set = {str(pmid) for pmid in retrieved_pmids}
    if parsed is not None:
        answer = str(parsed.get("answer") or "").strip()
        evidence_summary = str(parsed.get("evidence_summary") or "").strip()
        used_raw = parsed.get("used_pmids") or []
        if isinstance(used_raw, (str, bytes)):
            used_raw = [used_raw]
        used_pmids = []
        if isinstance(used_raw, Iterable):
            for pmid in used_raw:
                value = str(pmid).strip()
                if value and (not retrieved_set or value in retrieved_set):
                    used_pmids.append(value)
        return {
            "answer": answer
            or "The retrieved evidence was insufficient to answer the query.",
            "evidence_summary": evidence_summary,
            "used_pmids": used_pmids,
            "valid": bool(answer),
            "parse_status": "valid_json" if answer else "invalid",
            "parse_error": None if answer else "JSON answer was empty",
        }

    raw = str(raw_response or "").strip()
    return {
        "answer": raw
        or "The retrieved evidence was insufficient to answer the query.",
        "evidence_summary": "",
        "used_pmids": [],
        "valid": bool(raw),
        "parse_status": "recovered_text" if raw else "invalid",
        "parse_error": "response was not valid JSON",
    }


def answer_followup_query_with_pubmed_rag(
    query: str,
    pubmed_retriever: Any,
    llm: BaseLLMClient,
    k: int,
    temperature: float,
    max_tokens: int = 512,
    deduplicate_pmids: bool = True,
    max_chars_per_snippet: int = 1200,
    max_context_chars: int = 8000,
    llm_cache: Optional[_LLMCallCache] = None,
) -> Dict[str, Any]:
    """Retrieve PubMed evidence for a follow-up query and answer it concisely."""
    started = time.perf_counter()
    query = str(query or "").strip()
    if not query:
        raise ValueError("follow-up query must not be empty")
    llm_cache = llm_cache or _LLMCallCache(enabled=False, cache_dir="cache/unused")

    retrieval_started = time.perf_counter()
    raw_retrieval = _retrieve_with_k(pubmed_retriever, query, k=k)
    retrieval_latency = time.perf_counter() - retrieval_started
    retrieval = _deduplicate_retrieval(
        raw_retrieval, deduplicate_pmids=deduplicate_pmids
    )[:k]
    retrieved_pmids = [str(item.get("id") or "") for item in retrieval]
    retrieved_titles = [str(item.get("title") or "") for item in retrieval]

    if not retrieval:
        return {
            "query": query,
            "retrieval": [],
            "retrieved_pmids": [],
            "retrieved_titles": [],
            "answer": "No relevant PubMed evidence was retrieved.",
            "evidence_summary": "",
            "used_pmids": [],
            "raw_model_output": "",
            "prompt": "",
            "valid": True,
            "parse_status": "no_retrieval",
            "parse_error": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached": False,
            "retrieval_latency_seconds": retrieval_latency,
            "latency_seconds": time.perf_counter() - started,
        }

    evidence_context = _format_pubmed_evidence(
        retrieval,
        max_chars_per_snippet=max_chars_per_snippet,
        max_context_chars=max_context_chars,
    )
    prompt = build_followup_answer_prompt(query=query, evidence_context=evidence_context)
    call = llm_cache.chat(
        llm=llm,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stage="followup_answering",
    )
    parsed = _parse_followup_answer_response(
        call["raw_response"], retrieved_pmids=retrieved_pmids
    )
    if not parsed["used_pmids"] and parsed["valid"]:
        parsed["used_pmids"] = retrieved_pmids

    return {
        "query": query,
        "retrieval": retrieval,
        "retrieved_pmids": retrieved_pmids,
        "retrieved_titles": retrieved_titles,
        "answer": parsed["answer"],
        "evidence_summary": parsed["evidence_summary"],
        "used_pmids": parsed["used_pmids"],
        "raw_model_output": call["raw_response"],
        "prompt": prompt,
        "valid": parsed["valid"],
        "parse_status": parsed["parse_status"],
        "parse_error": parsed["parse_error"],
        "input_tokens": call["input_tokens"],
        "output_tokens": call["output_tokens"],
        "cached": call["cached"],
        "retrieval_latency_seconds": retrieval_latency,
        "latency_seconds": time.perf_counter() - started,
    }


def build_final_answer_prompt(
    question: str,
    options: Mapping[str, str],
    qa_history: Sequence[Mapping[str, Any]],
    dataset_name: str,
    max_history_items: Optional[int] = None,
) -> str:
    dataset_key = str(dataset_name or "").strip().lower()
    if dataset_key == "pubmedqa":
        schema = """{
  "answer_choice": "yes/no/maybe",
  "rationale": "Concise rationale based on the query-answer history."
}"""
    else:
        schema = """{
  "answer_choice": "A",
  "answer_text": "...",
  "rationale": "Concise rationale based on the query-answer history."
}"""

    return f"""You are answering a closed-set biomedical question using an iterative PubMed-RAG information-seeking history.

Original question:
{question}

Candidate answers:
{_format_options(options)}

Information-seeking history:
{_format_history(qa_history, max_items=max_history_items)}

Based on the question, candidate answers, and the information-seeking history, choose the single best answer from the candidate set.

Important requirements:
1. You must choose only one answer from the provided candidate set.
2. Do not output an answer outside the candidate set.
3. If the evidence is incomplete, still choose the best-supported candidate.
4. Do not mention ground-truth labels.
5. Return JSON only.

Return this schema:
{schema}"""


def _normalize_choice(
    raw_answer: Any,
    valid_labels: Sequence[str],
    options: Mapping[str, str],
    dataset_name: str,
) -> str:
    value = str(raw_answer or "").strip()
    if not value:
        return ""

    label_map = {str(label).strip().casefold(): str(label) for label in valid_labels}
    direct = label_map.get(value.casefold())
    if direct:
        return direct

    stripped = re.sub(
        r"^(?:answer|final answer|answer_choice|selected option|choice)\s*[:=]\s*",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    stripped = stripped.strip(".()[]{}\"'")
    direct = label_map.get(stripped.casefold())
    if direct:
        return direct

    label_prefix = re.match(r"^\s*([A-Za-z]+)\s*[\).:\-]\s*", stripped)
    if label_prefix:
        direct = label_map.get(label_prefix.group(1).casefold())
        if direct:
            return direct

    value_folded = stripped.casefold()
    for label, option_text in options.items():
        option_folded = str(option_text or "").strip().casefold()
        if option_folded and option_folded == value_folded:
            return str(label)

    dataset_key = str(dataset_name or "").strip().lower()
    if dataset_key == "pubmedqa":
        match = re.search(r"\b(yes|no|maybe)\b", value, flags=re.IGNORECASE)
        if match:
            return label_map.get(match.group(1).casefold(), "")
        return ""

    escaped_labels = [re.escape(str(label)) for label in valid_labels]
    if escaped_labels:
        pattern = r"\b(" + "|".join(escaped_labels) + r")\b"
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return label_map.get(match.group(1).casefold(), "")
    return ""


def _parse_final_answer_response(
    raw_response: Any,
    valid_labels: Sequence[str],
    options: Mapping[str, str],
    dataset_name: str,
) -> Dict[str, Any]:
    parsed = _first_json_object(raw_response)
    rationale = ""
    answer_text = ""
    if parsed is not None:
        rationale = str(parsed.get("rationale") or "").strip()
        answer_text = str(parsed.get("answer_text") or "").strip()
        answer_choice = _normalize_choice(
            parsed.get("answer_choice")
            or parsed.get("answer")
            or parsed.get("final_answer"),
            valid_labels=valid_labels,
            options=options,
            dataset_name=dataset_name,
        )
        if not answer_choice and answer_text:
            answer_choice = _normalize_choice(
                answer_text,
                valid_labels=valid_labels,
                options=options,
                dataset_name=dataset_name,
            )
        if answer_choice:
            return {
                "answer_choice": answer_choice,
                "answer_text": answer_text or options.get(answer_choice, answer_choice),
                "rationale": rationale,
                "valid": True,
                "parse_status": "valid_json",
                "parse_error": None,
            }
        return {
            "answer_choice": "",
            "answer_text": answer_text,
            "rationale": rationale,
            "valid": False,
            "parse_status": "invalid",
            "parse_error": "JSON answer_choice is not one of the candidate labels",
        }

    recovered = _normalize_choice(
        raw_response,
        valid_labels=valid_labels,
        options=options,
        dataset_name=dataset_name,
    )
    if recovered:
        return {
            "answer_choice": recovered,
            "answer_text": options.get(recovered, recovered),
            "rationale": "",
            "valid": True,
            "parse_status": "recovered_answer",
            "parse_error": "response was not valid JSON",
        }
    return {
        "answer_choice": "",
        "answer_text": "",
        "rationale": "",
        "valid": False,
        "parse_status": "invalid",
        "parse_error": "could not parse a candidate label",
    }


def generate_final_answer_from_history(
    question: str,
    options: Mapping[str, Any] | Sequence[Any],
    qa_history: List[Mapping[str, Any]],
    dataset_name: str,
    llm: BaseLLMClient,
    temperature: float,
    max_tokens: int = 512,
    max_history_items: int = 8,
    llm_cache: Optional[_LLMCallCache] = None,
) -> Dict[str, Any]:
    """Choose the final closed-set answer from the accumulated QA history."""
    normalized_options = _normalize_options(options, dataset_name=dataset_name)
    valid_labels = list(normalized_options.keys())
    llm_cache = llm_cache or _LLMCallCache(enabled=False, cache_dir="cache/unused")
    prompt = build_final_answer_prompt(
        question=question,
        options=normalized_options,
        qa_history=qa_history,
        dataset_name=dataset_name,
        max_history_items=max_history_items,
    )
    call = llm_cache.chat(
        llm=llm,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stage="final_answer",
    )
    parsed = _parse_final_answer_response(
        call["raw_response"],
        valid_labels=valid_labels,
        options=normalized_options,
        dataset_name=dataset_name,
    )
    return {
        **parsed,
        "raw_model_output": call["raw_response"],
        "prompt": prompt,
        "cached": call["cached"],
        "latency_seconds": call["latency_seconds"],
        "input_tokens": call["input_tokens"],
        "output_tokens": call["output_tokens"],
    }


def run_i_medrag_pubmed(
    question: str,
    options: Mapping[str, Any] | Sequence[Any],
    dataset_name: str,
    llm: BaseLLMClient,
    pubmed_retriever: Any,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run i-MedRAG-PubMed on one closed-set biomedical QA sample."""
    started = time.perf_counter()
    cfg = _merge_config(config)
    normalized_options = _normalize_options(options, dataset_name=dataset_name)
    llm_cache = _LLMCallCache(
        enabled=cfg["cache_llm_outputs"],
        cache_dir=cfg["llm_cache_dir"],
    )

    qa_history: List[Dict[str, Any]] = []
    rounds: List[Dict[str, Any]] = []
    seen_queries = set()
    token_usage = _empty_usage()
    num_pubmed_queries = 0
    num_retrieved_docs = 0
    unique_pmids = set()

    for round_id in range(1, cfg["n_rounds"] + 1):
        query_generation = _generate_followup_queries_with_metadata(
            question=question,
            options=normalized_options,
            qa_history=qa_history,
            llm=llm,
            n_queries=cfg["n_queries"],
            temperature=cfg["temperature_query"],
            max_tokens=cfg["max_tokens_query"],
            max_history_items=cfg["max_history_items"],
            llm_cache=llm_cache,
        )
        _add_usage(
            token_usage,
            "followup_query_generation",
            query_generation["input_tokens"],
            query_generation["output_tokens"],
        )

        round_queries: List[str] = []
        for query in query_generation["queries"]:
            normalized = _normalize_query(query)
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            round_queries.append(str(query).strip())
            if len(round_queries) >= cfg["n_queries"]:
                break

        if not round_queries:
            fallback = _fallback_pubmed_query(question)
            normalized = _normalize_query(fallback)
            if normalized and normalized not in seen_queries:
                seen_queries.add(normalized)
                round_queries = [fallback]

        round_log: Dict[str, Any] = {
            "round_id": round_id,
            "query_generation": {
                "raw_model_output": query_generation["raw_model_output"],
                "cached": query_generation["cached"],
                "latency_seconds": query_generation["latency_seconds"],
            },
            "queries": [],
        }

        for query in round_queries:
            followup = answer_followup_query_with_pubmed_rag(
                query=query,
                pubmed_retriever=pubmed_retriever,
                llm=llm,
                k=cfg["k_per_query"],
                temperature=cfg["temperature_answer"],
                max_tokens=cfg["max_tokens_answer"],
                deduplicate_pmids=cfg["deduplicate_pmids"],
                max_chars_per_snippet=cfg["max_chars_per_snippet"],
                max_context_chars=cfg["max_context_chars"],
                llm_cache=llm_cache,
            )
            _add_usage(
                token_usage,
                "followup_answering",
                followup["input_tokens"],
                followup["output_tokens"],
            )
            num_pubmed_queries += 1
            num_retrieved_docs += len(followup["retrieval"])
            unique_pmids.update(followup["retrieved_pmids"])

            history_item = {
                "query": query,
                "answer": followup["answer"],
                "evidence_summary": followup.get("evidence_summary", ""),
                "pmids": followup.get("used_pmids") or followup["retrieved_pmids"],
            }
            qa_history.append(history_item)
            round_log["queries"].append(
                {
                    "query": query,
                    "retrieved_pmids": followup["retrieved_pmids"],
                    "retrieved_titles": followup["retrieved_titles"],
                    "followup_answer": followup["answer"],
                    "evidence_summary": followup.get("evidence_summary", ""),
                    "used_pmids": followup.get("used_pmids", []),
                    "retrieval": followup["retrieval"],
                    "raw_model_output": followup["raw_model_output"],
                    "parse_status": followup["parse_status"],
                    "latency_seconds": followup["latency_seconds"],
                }
            )
        rounds.append(round_log)

    final_answer = generate_final_answer_from_history(
        question=question,
        options=normalized_options,
        qa_history=qa_history,
        dataset_name=dataset_name,
        llm=llm,
        temperature=cfg["temperature_final"],
        max_tokens=cfg["max_tokens_final"],
        max_history_items=cfg["max_history_items"],
        llm_cache=llm_cache,
    )
    _add_usage(
        token_usage,
        "final_answer",
        final_answer["input_tokens"],
        final_answer["output_tokens"],
    )

    latency = time.perf_counter() - started
    return {
        "method": METHOD_NAME,
        "dataset": dataset_name,
        "question": question,
        "options": normalized_options,
        "config": cfg,
        "rounds": rounds,
        "qa_history": qa_history,
        "final_prediction": final_answer["answer_choice"],
        "final_answer_text": final_answer["answer_text"],
        "final_rationale": final_answer["rationale"],
        "raw_final_output": final_answer["raw_model_output"],
        "answer": final_answer["answer_choice"],
        "rationale": final_answer["rationale"],
        "valid": final_answer["valid"],
        "parse_status": final_answer["parse_status"],
        "parse_error": final_answer["parse_error"],
        "num_followup_rounds": len(rounds),
        "num_followup_queries": len(qa_history),
        "num_pubmed_queries": num_pubmed_queries,
        "num_retrieved_docs": num_retrieved_docs,
        "num_unique_retrieved_docs": len(unique_pmids),
        "input_tokens": token_usage["input_tokens"],
        "output_tokens": token_usage["output_tokens"],
        "token_usage": token_usage,
        "latency": latency,
        "latency_seconds": latency,
    }


class IMedRAGPubMedBaseline:
    """Class wrapper for the i-MedRAG-PubMed baseline."""

    def __init__(
        self,
        llm: Optional[BaseLLMClient] = None,
        pubmed_retriever: Optional[Any] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = _merge_config(config)
        self.llm = llm or get_llm_client()
        if pubmed_retriever is None:
            self.pubmed_retriever = PubMedBM25Retriever(
                top_k=self.config["k_per_query"],
                candidate_k=max(self.config["candidate_k"], self.config["k_per_query"]),
                max_abstract_chars=self.config["max_abstract_chars"],
                cache_dir=(
                    self.config["retrieval_cache_dir"]
                    if self.config["cache_retrieval"]
                    else None
                ),
            )
        else:
            self.pubmed_retriever = pubmed_retriever

    def answer(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        return run_i_medrag_pubmed(
            question=str(sample.get("question") or sample.get("QUESTION") or ""),
            options=sample.get("options") or sample.get("OPTIONS") or {},
            dataset_name=str(sample.get("dataset") or sample.get("dataset_name") or ""),
            llm=self.llm,
            pubmed_retriever=self.pubmed_retriever,
            config=self.config,
        )

    def generate_followup_queries(
        self,
        question: str,
        options: Mapping[str, Any] | Sequence[Any],
        qa_history: List[Mapping[str, Any]],
    ) -> List[str]:
        return generate_followup_queries(
            question=question,
            options=options,
            qa_history=qa_history,
            llm=self.llm,
            n_queries=self.config["n_queries"],
            temperature=self.config["temperature_query"],
            max_tokens=self.config["max_tokens_query"],
        )

    def answer_followup_query(self, query: str) -> Dict[str, Any]:
        return answer_followup_query_with_pubmed_rag(
            query=query,
            pubmed_retriever=self.pubmed_retriever,
            llm=self.llm,
            k=self.config["k_per_query"],
            temperature=self.config["temperature_answer"],
            max_tokens=self.config["max_tokens_answer"],
            deduplicate_pmids=self.config["deduplicate_pmids"],
            max_chars_per_snippet=self.config["max_chars_per_snippet"],
            max_context_chars=self.config["max_context_chars"],
        )

    def generate_final_answer(
        self,
        question: str,
        options: Mapping[str, Any] | Sequence[Any],
        qa_history: List[Mapping[str, Any]],
        dataset_name: str,
    ) -> Dict[str, Any]:
        return generate_final_answer_from_history(
            question=question,
            options=options,
            qa_history=qa_history,
            dataset_name=dataset_name,
            llm=self.llm,
            temperature=self.config["temperature_final"],
            max_tokens=self.config["max_tokens_final"],
            max_history_items=self.config["max_history_items"],
        )
