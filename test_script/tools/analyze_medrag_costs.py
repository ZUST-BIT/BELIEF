"""Summarize MedRAG and i-MedRAG result logs with the cost benchmark schema.

The original cost benchmark in this repository is PubMedQA-only and reports
per-method accuracy, latency, token usage, and LLM call counts. This script
keeps that reporting shape while reading already generated baseline logs.

Examples:
    python test_script/tools/analyze_medrag_costs.py
    python test_script/tools/analyze_medrag_costs.py --dataset pubmedqa
    python test_script/tools/analyze_medrag_costs.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_OUTPUT_DIR = Path("TEST_RESULTS/cost_benchmark")

DATASET_LABELS = {
    "pubmedqa": ["yes", "no", "maybe"],
    "medqa": ["A", "B", "C", "D"],
    "medmcqa": ["A", "B", "C", "D"],
}

METHOD_CONFIGS = {
    "medrag": {
        "summary_name": "medrag_pubmed_online_bm25",
        "display_name": "MedRAG-PubMed",
        "root": Path("TEST_RESULTS/medrag"),
    },
    "imedrag": {
        "summary_name": "i-MedRAG-PubMed",
        "display_name": "i-MedRAG-PubMed",
        "root": Path("TEST_RESULTS/imedrag_pubmed"),
    },
}


def estimate_tokens(text: Any) -> int:
    value = str(text or "")
    if not value:
        return 0
    return max(1, math.ceil(len(value) / 4))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def safe_percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return ordered[int(k)]
    return ordered[lower] * (upper - k) + ordered[upper] * (k - lower)


def rounded(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def read_jsonl(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    summary_path = path.with_suffix(".summary.json")
    meta: Dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        meta = dict(payload.get("meta") or {})
    return meta, rows


def read_result_file(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".jsonl":
        meta, rows = read_jsonl(path)
        return {"meta": meta, "results": rows}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"result file is not a JSON object: {path}")
    return payload


def result_count(path: Path) -> int:
    try:
        payload = read_result_file(path)
    except Exception:
        return 0
    results = payload.get("results")
    return len(results) if isinstance(results, list) else 0


def latest_result_file(method_key: str, dataset: str, min_samples: int) -> Path:
    root = METHOD_CONFIGS[method_key]["root"] / dataset
    if not root.exists():
        raise FileNotFoundError(f"result directory does not exist: {root}")

    candidates = [
        path
        for path in root.glob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl"}
        and "summary" not in path.name.lower()
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for path in candidates:
        if result_count(path) >= min_samples:
            return path

    raise FileNotFoundError(
        f"no {method_key} result file with at least {min_samples} rows under {root}"
    )


def as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def count_imedrag_llm_calls(result: Mapping[str, Any]) -> int:
    if result.get("error"):
        return 0

    calls = 0
    for round_item in result.get("rounds") or []:
        if not isinstance(round_item, Mapping):
            continue
        query_generation = round_item.get("query_generation") or {}
        if isinstance(query_generation, Mapping) and (
            query_generation.get("raw_model_output")
            or is_number(query_generation.get("latency_seconds"))
        ):
            calls += 1

        for query_item in round_item.get("queries") or []:
            if isinstance(query_item, Mapping) and query_item.get("raw_model_output"):
                calls += 1

    if result.get("raw_final_output"):
        calls += 1
    return calls


def count_cached_imedrag_calls(result: Mapping[str, Any]) -> int:
    cached = 0
    for round_item in result.get("rounds") or []:
        if not isinstance(round_item, Mapping):
            continue
        query_generation = round_item.get("query_generation") or {}
        if isinstance(query_generation, Mapping) and query_generation.get("cached"):
            cached += 1
        for query_item in round_item.get("queries") or []:
            if isinstance(query_item, Mapping) and query_item.get("cached"):
                cached += 1
    return cached


def medrag_cost_row(result: Mapping[str, Any], labels: Sequence[str]) -> Dict[str, Any]:
    prompt_tokens = estimate_tokens(result.get("prompt"))
    completion_tokens = estimate_tokens(result.get("raw_model_output"))
    error = result.get("error")
    predicted = str(result.get("predicted") or result.get("answer") or "")
    valid = bool(result.get("valid")) or predicted in labels
    llm_calls = 1 if not error and (result.get("prompt") or result.get("raw_model_output") is not None) else 0
    retrieval = result.get("retrieval") or []
    elapsed = result.get("latency_seconds")

    return {
        "sample_id": result.get("sample_id"),
        "ground_truth": result.get("gold", ""),
        "prediction": predicted,
        "valid": valid,
        "is_correct": bool(result.get("is_correct")),
        "elapsed_s": float(elapsed) if is_number(elapsed) else None,
        "token_stats": {
            "llm_calls": llm_calls,
            "llm_calls_ok": llm_calls if not error else 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_calls": llm_calls,
            "llm_elapsed_s": float(elapsed) if is_number(elapsed) else None,
        },
        "retrieval_stats": {
            "pubmed_queries": 1 if not error else 0,
            "retrieved_docs": len(retrieval) if isinstance(retrieval, list) else 0,
            "unique_retrieved_docs": len(
                {
                    str(item.get("pmid") or item.get("id"))
                    for item in retrieval
                    if isinstance(item, Mapping) and (item.get("pmid") or item.get("id"))
                }
            )
            if isinstance(retrieval, list)
            else 0,
        },
        "error": error,
    }


def imedrag_cost_row(result: Mapping[str, Any], labels: Sequence[str]) -> Dict[str, Any]:
    token_usage = result.get("token_usage") or {}
    prompt_tokens = as_int(result.get("input_tokens", token_usage.get("input_tokens")))
    completion_tokens = as_int(result.get("output_tokens", token_usage.get("output_tokens")))
    elapsed = result.get("latency_seconds", result.get("latency"))
    error = result.get("error")
    predicted = str(result.get("predicted") or result.get("final_prediction") or result.get("answer") or "")
    valid = bool(result.get("valid")) or predicted in labels
    llm_calls = count_imedrag_llm_calls(result)

    return {
        "sample_id": result.get("sample_id"),
        "ground_truth": result.get("gold", ""),
        "prediction": predicted,
        "valid": valid,
        "is_correct": bool(result.get("is_correct")),
        "elapsed_s": float(elapsed) if is_number(elapsed) else None,
        "token_stats": {
            "llm_calls": llm_calls,
            "llm_calls_ok": llm_calls if not error else 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_calls": llm_calls if token_usage.get("estimated", True) else 0,
            "llm_elapsed_s": float(elapsed) if is_number(elapsed) else None,
        },
        "retrieval_stats": {
            "pubmed_queries": as_int(result.get("num_pubmed_queries")),
            "retrieved_docs": as_int(result.get("num_retrieved_docs")),
            "unique_retrieved_docs": as_int(result.get("num_unique_retrieved_docs")),
        },
        "cached_llm_calls": count_cached_imedrag_calls(result),
        "error": error,
    }


def rows_for_method(method_key: str, payload: Mapping[str, Any], labels: Sequence[str]) -> List[Dict[str, Any]]:
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise ValueError("result payload does not contain a results list")

    if method_key == "medrag":
        return [medrag_cost_row(result, labels) for result in results if isinstance(result, Mapping)]
    if method_key == "imedrag":
        return [imedrag_cost_row(result, labels) for result in results if isinstance(result, Mapping)]
    raise ValueError(f"unsupported method: {method_key}")


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    elapsed = [float(row["elapsed_s"]) for row in rows if is_number(row.get("elapsed_s"))]
    total_tokens = [as_int(row.get("token_stats", {}).get("total_tokens")) for row in rows]
    prompt_tokens = [as_int(row.get("token_stats", {}).get("prompt_tokens")) for row in rows]
    completion_tokens = [as_int(row.get("token_stats", {}).get("completion_tokens")) for row in rows]
    llm_calls = [as_int(row.get("token_stats", {}).get("llm_calls")) for row in rows]
    pubmed_queries = [as_int(row.get("retrieval_stats", {}).get("pubmed_queries")) for row in rows]
    retrieved_docs = [as_int(row.get("retrieval_stats", {}).get("retrieved_docs")) for row in rows]
    cached_calls = [as_int(row.get("cached_llm_calls")) for row in rows]

    attempted = len(rows)
    succeeded = sum(1 for row in rows if row.get("error") is None)
    valid_pred_rows = [row for row in rows if row.get("valid")]
    correct = sum(1 for row in valid_pred_rows if row.get("is_correct") is True)
    errors = attempted - succeeded

    return {
        "attempted": attempted,
        "succeeded": succeeded,
        "errors": errors,
        "valid_predictions": len(valid_pred_rows),
        "invalid_predictions": attempted - len(valid_pred_rows),
        "correct": correct,
        "accuracy": round(correct / attempted, 6) if attempted else 0.0,
        "latency_available": bool(elapsed),
        "total_elapsed_s": rounded(sum(elapsed), 4) if elapsed else None,
        "avg_elapsed_s": rounded(statistics.mean(elapsed), 4) if elapsed else None,
        "p50_elapsed_s": rounded(safe_percentile(elapsed, 0.5), 4),
        "p95_elapsed_s": rounded(safe_percentile(elapsed, 0.95), 4),
        "total_prompt_tokens": int(sum(prompt_tokens)),
        "total_completion_tokens": int(sum(completion_tokens)),
        "total_tokens": int(sum(total_tokens)),
        "avg_total_tokens_per_q": round(statistics.mean(total_tokens), 2) if total_tokens else 0.0,
        "avg_prompt_tokens_per_q": round(statistics.mean(prompt_tokens), 2) if prompt_tokens else 0.0,
        "avg_completion_tokens_per_q": round(statistics.mean(completion_tokens), 2) if completion_tokens else 0.0,
        "total_llm_calls": int(sum(llm_calls)),
        "avg_llm_calls_per_q": round(statistics.mean(llm_calls), 3) if llm_calls else 0.0,
        "estimated_token_calls": int(
            sum(as_int(row.get("token_stats", {}).get("estimated_calls")) for row in rows)
        ),
        "total_cached_llm_calls": int(sum(cached_calls)),
        "avg_cached_llm_calls_per_q": round(statistics.mean(cached_calls), 3) if cached_calls else 0.0,
        "total_pubmed_queries": int(sum(pubmed_queries)),
        "avg_pubmed_queries_per_q": round(statistics.mean(pubmed_queries), 3) if pubmed_queries else 0.0,
        "total_retrieved_docs": int(sum(retrieved_docs)),
        "avg_retrieved_docs_per_q": round(statistics.mean(retrieved_docs), 3) if retrieved_docs else 0.0,
    }


def build_detail_records(method_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    sample_order: List[str] = []
    details: Dict[str, Dict[str, Any]] = {}

    for method_name, rows in method_rows.items():
        for index, row in enumerate(rows):
            sample_id = str(row.get("sample_id") or index)
            if sample_id not in details:
                details[sample_id] = {
                    "sample_id": row.get("sample_id"),
                    "ground_truth": row.get("ground_truth", ""),
                    "methods": {},
                }
                sample_order.append(sample_id)
            details[sample_id]["methods"][method_name] = row

    return [details[sample_id] for sample_id in sample_order]


def analyze_dataset(
    dataset: str,
    medrag_file: Optional[Path],
    imedrag_file: Optional[Path],
    min_samples: int,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    labels = DATASET_LABELS[dataset]
    selected_files = {
        "medrag": medrag_file or latest_result_file("medrag", dataset, min_samples),
        "imedrag": imedrag_file or latest_result_file("imedrag", dataset, min_samples),
    }

    summary: Dict[str, Any] = {}
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    for method_key, path in selected_files.items():
        payload = read_result_file(path)
        method_name = METHOD_CONFIGS[method_key]["summary_name"]
        rows = rows_for_method(method_key, payload, labels)
        stats = summarize_rows(rows)
        stats.update(
            {
                "dataset": dataset,
                "display_name": METHOD_CONFIGS[method_key]["display_name"],
                "source_file": str(path),
                "source_meta": payload.get("meta", {}),
            }
        )
        summary[method_name] = stats
        all_rows[method_name] = rows
    return summary, all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MedRAG/i-MedRAG cost from existing result logs"
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_LABELS) + ["all"],
        default="pubmedqa",
        help="Dataset to analyze. Defaults to PubMedQA to match benchmark_pubmedqa_costs.py.",
    )
    parser.add_argument("--medrag-file", type=Path, default=None)
    parser.add_argument("--imedrag-file", type=Path, default=None)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_samples <= 0:
        raise ValueError("--min-samples must be positive")
    if args.dataset == "all" and (args.medrag_file or args.imedrag_file):
        raise ValueError("--medrag-file/--imedrag-file can only be used with one dataset")

    datasets = sorted(DATASET_LABELS) if args.dataset == "all" else [args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_summary: Dict[str, Any] = {}
    combined_details: Dict[str, Any] = {}

    for dataset in datasets:
        summary, rows = analyze_dataset(
            dataset=dataset,
            medrag_file=args.medrag_file,
            imedrag_file=args.imedrag_file,
            min_samples=args.min_samples,
        )
        combined_summary[dataset] = summary
        combined_details[dataset] = build_detail_records(rows)

    meta = {
        "timestamp": ts,
        "script": "test_script/tools/analyze_medrag_costs.py",
        "dataset": args.dataset,
        "datasets": datasets,
        "standard": "benchmark_pubmedqa_costs.py summary fields; token estimates use ceil(len(text)/4)",
        "notes": [
            "MedRAG historical logs do not contain per-question latency; elapsed fields are null for MedRAG unless supplied in the source log.",
            "i-MedRAG token counts are the baseline's stored estimates, including logical calls served from cache.",
        ],
    }
    payload = {"meta": meta, "summary": combined_summary}
    detail_payload = {"meta": meta, "results": combined_details}

    dataset_slug = args.dataset.lower()
    summary_path = args.output_dir / f"{dataset_slug}_medrag_cost_summary_{ts}.json"
    detail_path = args.output_dir / f"{dataset_slug}_medrag_cost_detail_{ts}.json"

    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    detail_path.write_text(json.dumps(detail_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved files:")
    print(f"- {detail_path}")
    print(f"- {summary_path}")
    print()
    for dataset in datasets:
        print(f"[{dataset}]")
        for method_name, stats in combined_summary[dataset].items():
            avg_time = stats["avg_elapsed_s"]
            avg_time_text = "NA" if avg_time is None else f"{avg_time:.3f}s"
            print(
                f"{method_name}: "
                f"acc={stats['accuracy']:.4f}, "
                f"avg_time={avg_time_text}, "
                f"avg_tokens={stats['avg_total_tokens_per_q']:.1f}, "
                f"avg_llm_calls={stats['avg_llm_calls_per_q']:.3f}, "
                f"avg_pubmed_q={stats['avg_pubmed_queries_per_q']:.3f}, "
                f"errors={stats['errors']}, "
                f"source={stats['source_file']}"
            )


if __name__ == "__main__":
    main()
