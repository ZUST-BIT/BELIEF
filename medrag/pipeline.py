"""Prompting, generation, and answer parsing for the MedRAG baseline."""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Dict, Mapping, Optional, Sequence

from llm_client import BaseLLMClient, get_llm_client

from .retrieval import PubMedBM25Retriever, format_evidence_context


METHOD_NAME = "MedRAG-PubMed"


def _estimate_tokens(text: Any) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(str(text)) / 4))


def _normalized_options(
    options: Optional[Mapping[str, Any]], dataset_name: str
) -> Dict[str, str]:
    if options:
        normalized = {
            str(label).strip(): str(text).strip()
            for label, text in options.items()
            if str(label).strip()
        }
        if normalized:
            return normalized
    if dataset_name.strip().lower() == "pubmedqa":
        return {"yes": "yes", "no": "no", "maybe": "maybe"}
    raise ValueError("multiple-choice examples must provide candidate options")


def build_medrag_prompt(
    question: str,
    options: Mapping[str, str],
    evidence_context: str,
    dataset_name: str,
) -> str:
    """Construct a closed-set MedRAG prompt with evidence before the question."""
    valid_labels = list(options.keys())
    option_lines = "\n".join(
        f"{label}. {text}" for label, text in options.items()
    )
    labels_json = json.dumps(valid_labels, ensure_ascii=False)
    return f"""You are a medical question-answering system using retrieved PubMed evidence.

Retrieved PubMed evidence (ranked):
{evidence_context}

Dataset: {dataset_name}

Question:
{question}

Candidate answers:
{option_lines}

Instructions:
1. Analyze the question step by step using the retrieved evidence. The evidence may be incomplete or irrelevant, so weigh it carefully and use medical knowledge only when necessary.
2. Select exactly one answer from these valid labels: {labels_json}.
3. Return only one JSON object. Do not use Markdown or add text outside the JSON.
4. Keep the rationale concise and evidence-based; do not provide a long hidden chain of thought.

Required output schema:
{{"rationale": "brief reasoning based on the retrieved PubMed snippets", "answer": "one valid label"}}"""


def _json_objects(text: str) -> Sequence[str]:
    objects = []
    start = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
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


def _normalize_label(raw_answer: Any, valid_labels: Sequence[str]) -> str:
    value = str(raw_answer or "").strip()
    if not value:
        return ""
    label_map = {str(label).strip().casefold(): str(label) for label in valid_labels}
    direct = label_map.get(value.casefold())
    if direct:
        return direct

    stripped = re.sub(
        r"^(?:answer|final answer|selected option)\s*[:=]\s*",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip().strip(".()[]{}\"'")
    return label_map.get(stripped.casefold(), "")


def parse_medrag_response(
    raw_response: Optional[str], valid_labels: Sequence[str]
) -> Dict[str, Any]:
    """Parse model output without guessing an answer from arbitrary rationale text."""
    raw = str(raw_response or "").strip()
    if not raw:
        return {
            "rationale": "",
            "answer": "",
            "valid": False,
            "parse_status": "invalid",
            "parse_error": "empty model response",
        }

    parsed: Optional[Dict[str, Any]] = None
    candidates = [raw] + list(_json_objects(raw))
    seen = set()
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        if cleaned in seen:
            continue
        seen.add(cleaned)
        try:
            value = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            parsed = value
            break

    if parsed is not None:
        answer = _normalize_label(parsed.get("answer", ""), valid_labels)
        rationale = str(parsed.get("rationale", "") or "").strip()
        if answer:
            return {
                "rationale": rationale,
                "answer": answer,
                "valid": True,
                "parse_status": "valid_json",
                "parse_error": None,
            }
        return {
            "rationale": rationale,
            "answer": "",
            "valid": False,
            "parse_status": "invalid",
            "parse_error": "JSON answer is not one of the candidate labels",
        }

    explicit = re.search(
        r"(?:^|\n)\s*(?:final\s+)?answer\s*[:=]\s*([^\n,}]+)",
        raw,
        flags=re.IGNORECASE,
    )
    recovered = _normalize_label(explicit.group(1), valid_labels) if explicit else ""
    if not recovered:
        recovered = _normalize_label(raw, valid_labels)
    if recovered:
        return {
            "rationale": "",
            "answer": recovered,
            "valid": True,
            "parse_status": "recovered_answer",
            "parse_error": "response was not valid JSON",
        }
    return {
        "rationale": "",
        "answer": "",
        "valid": False,
        "parse_status": "invalid",
        "parse_error": "could not parse a candidate label",
    }


class MedRAGPipeline:
    """End-to-end question-only PubMed retrieval and closed-set generation."""

    def __init__(
        self,
        retriever: Optional[PubMedBM25Retriever] = None,
        llm: Optional[BaseLLMClient] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        max_chars_per_snippet: int = 1800,
        max_context_chars: int = 10000,
    ) -> None:
        self.retriever = retriever or PubMedBM25Retriever()
        self.llm = llm or get_llm_client()
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_chars_per_snippet = int(max_chars_per_snippet)
        self.max_context_chars = int(max_context_chars)

    def run(
        self,
        question: str,
        options: Optional[Mapping[str, Any]],
        sample_id: Any,
        dataset_name: str,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        query = str(question or "").strip()
        if not query:
            raise ValueError("question must not be empty")
        normalized_options = _normalized_options(options, dataset_name)

        # The retriever receives only the question by construction.
        retrieval_started = time.perf_counter()
        retrieval = self.retriever.retrieve(query)
        retrieval_latency = time.perf_counter() - retrieval_started
        evidence_context = format_evidence_context(
            retrieval,
            max_chars_per_snippet=self.max_chars_per_snippet,
            max_context_chars=self.max_context_chars,
        )
        prompt = build_medrag_prompt(
            question=query,
            options=normalized_options,
            evidence_context=evidence_context,
            dataset_name=dataset_name,
        )
        llm_started = time.perf_counter()
        raw_response = self.llm.chat(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        llm_latency = time.perf_counter() - llm_started
        parsed = parse_medrag_response(raw_response, list(normalized_options.keys()))
        input_tokens = _estimate_tokens(prompt)
        output_tokens = _estimate_tokens(raw_response)
        latency = time.perf_counter() - started
        retrieved_ids = {
            str(item.get("id") or item.get("pmid") or "")
            for item in retrieval
            if isinstance(item, Mapping) and (item.get("id") or item.get("pmid"))
        }
        return {
            "method": METHOD_NAME,
            "sample_id": sample_id,
            "dataset": dataset_name,
            "question": query,
            "options": normalized_options,
            "retrieval_query": query,
            "retrieval": retrieval,
            "evidence_context": evidence_context,
            "prompt": prompt,
            "rationale": parsed["rationale"],
            "answer": parsed["answer"],
            "valid": parsed["valid"],
            "parse_status": parsed["parse_status"],
            "parse_error": parsed["parse_error"],
            "raw_model_output": raw_response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "by_stage": {
                    "answer_generation": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                },
            },
            "llm_calls": 1,
            "num_pubmed_queries": 1,
            "num_retrieved_docs": len(retrieval),
            "num_unique_retrieved_docs": len(retrieved_ids),
            "retrieval_latency_seconds": retrieval_latency,
            "llm_latency_seconds": llm_latency,
            "latency": latency,
            "latency_seconds": latency,
        }
