"""
PubMed-based MedRAG baseline for closed-set medical question answering.

Examples:
  python test_script/medrag_baseline.py --dataset pubmedqa --limit 20
  python test_script/medrag_baseline.py --dataset medqa --limit 100 --top-k 5
  python test_script/medrag_baseline.py --dataset medmcqa --offset 100 --limit 100

The retrieval path is deliberately question-only:
question -> online PubMed candidate pool -> local BM25 -> LLM answer selection.
Dataset contexts, candidate answers, and gold labels are never sent to retrieval.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tqdm import tqdm

from llm_client import get_llm_client
from medrag import MedRAGPipeline, PubMedBM25Retriever


DATASET_CONFIGS = {
    "pubmedqa": {
        "data_path": "datasets/pubmedqa.json",
        "format": "json",
        "labels": ["yes", "no", "maybe"],
        "default_limit": 500,
    },
    "medqa": {
        "data_path": "datasets/medqa.jsonl",
        "format": "jsonl",
        "labels": ["A", "B", "C", "D"],
        "default_limit": 500,
    },
    "medmcqa": {
        "data_path": "datasets/medmcqa.jsonl",
        "format": "jsonl",
        "labels": ["A", "B", "C", "D"],
        "default_limit": 1000,
    },
}


def _first_present(item: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


def _iter_json_records(raw: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw, list):
        for index, item in enumerate(raw):
            if isinstance(item, dict):
                record = dict(item)
                record.setdefault("_key", index)
                yield record
        return
    if isinstance(raw, dict):
        # Also accept an existing result wrapper if it is supplied intentionally.
        if isinstance(raw.get("data"), list):
            yield from _iter_json_records(raw["data"])
            return
        for key, item in raw.items():
            if isinstance(item, dict):
                record = dict(item)
                record.setdefault("_key", key)
                yield record


def load_dataset(
    path: str | Path,
    file_format: str,
    offset: int = 0,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load JSON-dict, JSON-list, or JSONL examples with deterministic slicing."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"dataset file does not exist: {data_path}")
    if offset < 0:
        raise ValueError("offset must not be negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    records: Iterable[Dict[str, Any]]
    if file_format == "jsonl" or data_path.suffix.lower() == ".jsonl":
        parsed: List[Dict[str, Any]] = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                parsed.append(value)
        records = parsed
    else:
        with data_path.open("r", encoding="utf-8") as handle:
            records = list(_iter_json_records(json.load(handle)))

    data = list(records)
    end = None if limit is None else offset + limit
    return data[offset:end]


def normalize_example(item: Mapping[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Map the repository datasets to the MedRAG input contract."""
    question = str(_first_present(item, ("question", "QUESTION"), "") or "").strip()
    if not question:
        raise ValueError("example has no question")
    sample_id = _first_present(
        item, ("sample_id", "realidx", "idx", "id", "_key"), "unknown"
    )

    if dataset_name == "pubmedqa":
        # PubMedQA CONTEXTS are intentionally ignored: this baseline uses only
        # evidence retrieved online from the question.
        options = {"yes": "yes", "no": "no", "maybe": "maybe"}
        gold_raw = _first_present(
            item, ("final_decision", "answer", "label", "gold"), ""
        )
    else:
        raw_options = item.get("options", {})
        if not isinstance(raw_options, dict) or not raw_options:
            raise ValueError("multiple-choice example has no options dictionary")
        options = {
            str(label).strip(): str(text).strip()
            for label, text in raw_options.items()
            if str(label).strip()
        }
        gold_raw = _first_present(
            item, ("answer_idx", "label", "gold", "answer"), ""
        )

    labels = list(options.keys())
    gold = normalize_gold_answer(gold_raw, labels, options, item)
    return {
        "sample_id": sample_id,
        "dataset": str(item.get("dataset") or dataset_name),
        "question": question,
        "options": options,
        "gold": gold,
    }


def normalize_gold_answer(
    raw: Any,
    valid_labels: Sequence[str],
    options: Mapping[str, str],
    item: Optional[Mapping[str, Any]] = None,
) -> str:
    label_map = {str(label).casefold(): str(label) for label in valid_labels}
    value = str(raw or "").strip()
    direct = label_map.get(value.casefold())
    if direct:
        return direct

    # Some MedMCQA exports encode the correct option as a zero-based integer.
    if value.isdigit():
        index = int(value)
        if 0 <= index < len(valid_labels):
            return str(valid_labels[index])

    answer_text = str((item or {}).get("answer", "") or value).strip().casefold()
    for label, option_text in options.items():
        if str(option_text).strip().casefold() == answer_text:
            return str(label)
    return ""


def compute_metrics(
    results: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> Dict[str, Any]:
    total = len(results)
    correct = sum(bool(result.get("is_correct")) for result in results)
    valid = sum(bool(result.get("valid")) for result in results)
    retrieval_counts = [
        int(result.get("num_retrieved_docs", len(result.get("retrieval", []) or [])) or 0)
        for result in results
    ]
    latency_values = [
        float(result.get("latency_seconds", 0.0) or 0.0)
        for result in results
        if result.get("error") is None
    ]
    input_tokens = [int(result.get("input_tokens", 0) or 0) for result in results]
    output_tokens = [int(result.get("output_tokens", 0) or 0) for result in results]
    total_tokens = [inp + out for inp, out in zip(input_tokens, output_tokens)]
    llm_calls = [int(result.get("llm_calls", 0) or 0) for result in results]
    if not any(llm_calls):
        llm_calls = [
            0
            if result.get("error")
            else (1 if (result.get("input_tokens") or result.get("output_tokens")) else 0)
            for result in results
        ]
    pubmed_queries = [
        int(result.get("num_pubmed_queries", 0) or 0)
        for result in results
    ]
    retrieved_docs = [
        int(result.get("num_retrieved_docs", len(result.get("retrieval", []) or [])) or 0)
        for result in results
    ]
    per_label: Dict[str, Dict[str, Any]] = {}
    f1_values: List[float] = []

    for label in labels:
        true_positive = sum(
            result.get("gold") == label and result.get("predicted") == label
            for result in results
        )
        false_positive = sum(
            result.get("gold") != label and result.get("predicted") == label
            for result in results
        )
        false_negative = sum(
            result.get("gold") == label and result.get("predicted") != label
            for result in results
        )
        support = sum(result.get("gold") == label for result in results)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_label[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    def average(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def percentile(values: Sequence[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        k = (len(ordered) - 1) * p
        lower = int(k)
        upper = min(lower + 1, len(ordered) - 1)
        weight = k - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    total_input_tokens = sum(input_tokens)
    total_output_tokens = sum(output_tokens)
    total_token_count = total_input_tokens + total_output_tokens
    total_llm_calls = sum(llm_calls)
    total_pubmed_queries = sum(pubmed_queries)
    total_retrieved_docs = sum(retrieved_docs)

    return {
        "total": total,
        "correct": correct,
        "valid_predictions": valid,
        "invalid_predictions": total - valid,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 6) if f1_values else 0.0,
        "no_retrieval": sum(count == 0 for count in retrieval_counts),
        "average_retrieved": (
            round(sum(retrieval_counts) / len(retrieval_counts), 6)
            if retrieval_counts
            else 0.0
        ),
        "total_latency_seconds": round(sum(latency_values), 4),
        "avg_latency_seconds": round(average(latency_values), 4),
        "p50_latency_seconds": round(percentile(latency_values, 0.5), 4),
        "p95_latency_seconds": round(percentile(latency_values, 0.95), 4),
        "total_input_tokens": int(total_input_tokens),
        "total_output_tokens": int(total_output_tokens),
        "total_tokens": int(total_token_count),
        "avg_input_tokens_per_q": round(average(input_tokens), 2),
        "avg_output_tokens_per_q": round(average(output_tokens), 2),
        "avg_total_tokens_per_q": round(average(total_tokens), 2),
        "total_llm_calls": int(total_llm_calls),
        "avg_llm_calls_per_q": round(average(llm_calls), 3),
        "total_pubmed_queries": int(total_pubmed_queries),
        "avg_pubmed_queries_per_q": round(average(pubmed_queries), 3),
        "total_retrieved_docs": int(total_retrieved_docs),
        "avg_retrieved_docs_per_q": round(average(retrieved_docs), 3),
        "per_label": per_label,
    }


class MedRAGBaselineEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = DATASET_CONFIGS[args.dataset]
        self.labels = list(self.config["labels"])
        self.results: List[Dict[str, Any]] = []
        self.error_count = 0

        retriever = PubMedBM25Retriever(
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            max_abstract_chars=args.max_abstract_chars,
            cache_dir=None if args.no_cache else args.cache_dir,
            refresh_cache=args.refresh_cache,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            request_timeout=args.request_timeout,
            retry=args.retries,
        )
        self.pipeline = MedRAGPipeline(
            retriever=retriever,
            llm=get_llm_client(),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_chars_per_snippet=args.max_chars_per_snippet,
            max_context_chars=args.max_context_chars,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / f"medrag_{args.dataset}_{timestamp}.json"

    def run_single(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        example: Optional[Dict[str, Any]] = None
        try:
            example = normalize_example(item, self.args.dataset)
            output = self.pipeline.run(
                question=example["question"],
                options=example["options"],
                sample_id=example["sample_id"],
                dataset_name=example["dataset"],
            )
            predicted = output.get("answer", "")
            return {
                **output,
                "gold": example["gold"],
                "predicted": predicted,
                "is_correct": bool(predicted) and predicted == example["gold"],
                "error": None,
            }
        except Exception as exc:
            self.error_count += 1
            return {
                "sample_id": (example or {}).get(
                    "sample_id", _first_present(item, ("realidx", "idx", "_key"), "unknown")
                ),
                "method": "MedRAG-PubMed",
                "dataset": self.args.dataset,
                "question": (example or {}).get(
                    "question", _first_present(item, ("question", "QUESTION"), "")
                ),
                "gold": (example or {}).get("gold", ""),
                "predicted": "",
                "rationale": "",
                "answer": "",
                "valid": False,
                "parse_status": "error",
                "parse_error": None,
                "retrieval": [],
                "raw_model_output": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "llm_calls": 0,
                "num_pubmed_queries": 0,
                "num_retrieved_docs": 0,
                "num_unique_retrieved_docs": 0,
                "retrieval_latency_seconds": 0.0,
                "llm_latency_seconds": 0.0,
                "latency": 0.0,
                "latency_seconds": 0.0,
                "is_correct": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _metadata(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": self.args.dataset,
            "method": "medrag_pubmed_online_bm25",
            "retrieval_query": "question_only",
            "retrieval_strategy": "online_pubmed_candidate_pool_then_local_bm25",
            "data_path": self.args.data_path,
            "offset": self.args.offset,
            "limit": self.args.limit,
            "top_k": self.args.top_k,
            "candidate_k": self.args.candidate_k,
            "bm25_k1": self.args.bm25_k1,
            "bm25_b": self.args.bm25_b,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "error_count": self.error_count,
        }

    def save(self) -> None:
        payload = {
            "meta": self._metadata(),
            "metrics": compute_metrics(self.results, self.labels),
            "results": self.results,
        }
        temporary_path = self.output_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.output_path)

    def run(self) -> Dict[str, Any]:
        data = load_dataset(
            self.args.data_path,
            self.config["format"],
            offset=self.args.offset,
            limit=self.args.limit,
        )
        print("=" * 80)
        print(f"MedRAG baseline — {self.args.dataset}")
        print("Retrieval: question-only online PubMed -> local BM25")
        print(f"Candidate pool: {self.args.candidate_k} | Top-k: {self.args.top_k}")
        print(f"Examples: {len(data)} | Output: {self.output_path}")
        print("=" * 80)

        try:
            for index, item in enumerate(tqdm(data, desc="MedRAG"), start=1):
                result = self.run_single(item)
                if self.args.omit_prompt:
                    result.pop("prompt", None)
                if self.args.omit_context:
                    result.pop("evidence_context", None)
                self.results.append(result)

                metrics = compute_metrics(self.results, self.labels)
                flag = "✓" if result["is_correct"] else "✗"
                tqdm.write(
                    f"[{index}/{len(data)}] {flag} "
                    f"GT={result.get('gold', ''):<8} "
                    f"Pred={result.get('predicted', ''):<8} "
                    f"Retrieved={result.get('num_retrieved_docs', len(result.get('retrieval', [])))} "
                    f"Acc={metrics['accuracy'] * 100:.1f}%"
                )
                if index % self.args.save_interval == 0:
                    self.save()
        except KeyboardInterrupt:
            self.save()
            print(f"\nInterrupted; partial results saved to {self.output_path}")
            raise

        self.save()
        metrics = compute_metrics(self.results, self.labels)
        print("\n" + "=" * 80)
        print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Invalid  : {metrics['invalid_predictions']}")
        print(f"No retrieval: {metrics['no_retrieval']}")
        print(f"Errors   : {self.error_count}")
        print(f"Saved    : {self.output_path}")
        print("=" * 80)
        return {"metrics": metrics, "output_path": str(self.output_path)}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PubMed online BM25 MedRAG baseline"
    )
    parser.add_argument(
        "--dataset", choices=sorted(DATASET_CONFIGS), default="pubmedqa"
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--max-abstract-chars", type=int, default=3000)
    parser.add_argument("--max-chars-per-snippet", type=int, default=1800)
    parser.add_argument("--max-context-chars", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--cache-dir", default="cache/medrag_pubmed")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--omit-prompt", action="store_true")
    parser.add_argument("--omit-context", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = DATASET_CONFIGS[args.dataset]
    if args.data_path is None:
        args.data_path = config["data_path"]
    if args.output_dir is None:
        args.output_dir = f"TEST_RESULTS/medrag/{args.dataset}"
    if args.limit is None:
        args.limit = config["default_limit"]
    if args.save_interval <= 0:
        parser.error("--save-interval must be greater than zero")

    evaluator = MedRAGBaselineEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
