"""Evaluation entrypoint for the i-MedRAG-PubMed baseline.

Examples:
  python test_script/imedrag_pubmed_baseline.py --dataset pubmedqa --limit 5
  python test_script/imedrag_pubmed_baseline.py --dataset medqa --limit 20
  python test_script/imedrag_pubmed_baseline.py --dataset all --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tqdm import tqdm

from config import LLM_BACKEND, MODEL_NAME, OLLAMA_MODEL, VLLM_MODEL
from llm_client import get_llm_client
from medrag import DEFAULT_I_MEDRAG_PUBMED_CONFIG, PubMedBM25Retriever, run_i_medrag_pubmed
from test_script.medrag_baseline import (
    DATASET_CONFIGS,
    _first_present,
    compute_metrics,
    load_dataset,
    normalize_example,
)


METHOD_NAME = "i-MedRAG-PubMed"


def _current_backbone_name() -> str:
    if LLM_BACKEND == "api":
        return MODEL_NAME
    if LLM_BACKEND == "ollama":
        return OLLAMA_MODEL
    if LLM_BACKEND == "vllm":
        return VLLM_MODEL
    return LLM_BACKEND


def build_baseline_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(DEFAULT_I_MEDRAG_PUBMED_CONFIG)
    config.update(
        {
            "n_rounds": args.n_rounds,
            "n_queries": args.n_queries,
            "k_per_query": args.k_per_query,
            "max_history_items": args.max_history_items,
            "temperature_query": args.temperature_query,
            "temperature_answer": args.temperature_answer,
            "temperature_final": args.temperature_final,
            "max_tokens_query": args.max_tokens_query,
            "max_tokens_answer": args.max_tokens_answer,
            "max_tokens_final": args.max_tokens_final,
            "deduplicate_pmids": not args.no_deduplicate_pmids,
            "cache_retrieval": not args.no_cache,
            "cache_llm_outputs": not args.no_llm_cache,
            "candidate_k": args.candidate_k,
            "max_abstract_chars": args.max_abstract_chars,
            "max_chars_per_snippet": args.max_chars_per_snippet,
            "max_context_chars": args.max_context_chars,
            "retrieval_cache_dir": args.cache_dir,
            "llm_cache_dir": args.llm_cache_dir,
        }
    )
    return config


class IMedRAGPubMedEvaluator:
    def __init__(self, args: argparse.Namespace, dataset: str) -> None:
        self.args = args
        self.dataset = dataset
        self.dataset_config = DATASET_CONFIGS[dataset]
        self.labels = list(self.dataset_config["labels"])
        self.results: List[Dict[str, Any]] = []
        self.error_count = 0
        self.backbone = _current_backbone_name()

        self.baseline_config = build_baseline_config(args)
        self.retriever = PubMedBM25Retriever(
            top_k=args.k_per_query,
            candidate_k=args.candidate_k,
            max_abstract_chars=args.max_abstract_chars,
            cache_dir=None if args.no_cache else args.cache_dir,
            refresh_cache=args.refresh_cache,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            request_timeout=args.request_timeout,
            retry=args.retries,
        )
        self.llm = get_llm_client()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output:
            self.output_path = Path(args.output)
        else:
            output_dir = Path(args.output_dir or f"TEST_RESULTS/imedrag_pubmed/{dataset}")
            self.output_path = output_dir / f"imedrag_pubmed_{dataset}_{timestamp}.json"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def data_path(self) -> str:
        return self.args.data_path or self.dataset_config["data_path"]

    @property
    def limit(self) -> int:
        return (
            self.args.limit
            if self.args.limit is not None
            else int(self.dataset_config["default_limit"])
        )

    def run_single(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        example: Optional[Dict[str, Any]] = None
        try:
            example = normalize_example(item, self.dataset)
            output = run_i_medrag_pubmed(
                question=example["question"],
                options=example["options"],
                dataset_name=example["dataset"],
                llm=self.llm,
                pubmed_retriever=self.retriever,
                config=self.baseline_config,
            )
            predicted = output.get("final_prediction", "")
            return {
                **output,
                "sample_id": example["sample_id"],
                "gold": example["gold"],
                "predicted": predicted,
                "is_correct": bool(predicted) and predicted == example["gold"],
                "error": None,
            }
        except Exception as exc:
            self.error_count += 1
            return {
                "method": METHOD_NAME,
                "sample_id": (example or {}).get(
                    "sample_id",
                    _first_present(item, ("realidx", "idx", "id", "_key"), "unknown"),
                ),
                "dataset": self.dataset,
                "question": (example or {}).get(
                    "question", _first_present(item, ("question", "QUESTION"), "")
                ),
                "options": (example or {}).get("options", {}),
                "gold": (example or {}).get("gold", ""),
                "predicted": "",
                "final_prediction": "",
                "final_rationale": "",
                "answer": "",
                "valid": False,
                "parse_status": "error",
                "parse_error": None,
                "rounds": [],
                "qa_history": [],
                "num_pubmed_queries": 0,
                "num_retrieved_docs": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_seconds": 0.0,
                "is_correct": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _metadata(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": self.dataset,
            "method": METHOD_NAME,
            "backbone": self.backbone,
            "llm_backend": LLM_BACKEND,
            "retrieval_strategy": "iterative_followup_queries_online_pubmed_bm25",
            "data_path": self.data_path,
            "offset": self.args.offset,
            "limit": self.limit,
            "candidate_k": self.args.candidate_k,
            "bm25_k1": self.args.bm25_k1,
            "bm25_b": self.args.bm25_b,
            "error_count": self.error_count,
            "baseline_config": self.baseline_config,
        }

    def _summary_path(self) -> Path:
        if self.output_path.suffix.lower() == ".jsonl":
            return self.output_path.with_suffix(".summary.json")
        return self.output_path

    def save(self) -> None:
        metrics = compute_metrics(self.results, self.labels)
        metadata = self._metadata()

        if self.output_path.suffix.lower() == ".jsonl":
            temporary_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
            lines = [
                json.dumps(result, ensure_ascii=False, sort_keys=False)
                for result in self.results
            ]
            temporary_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            temporary_path.replace(self.output_path)

            summary_path = self._summary_path()
            summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
            summary_tmp.write_text(
                json.dumps(
                    {"meta": metadata, "metrics": metrics, "output_jsonl": str(self.output_path)},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            summary_tmp.replace(summary_path)
            return

        payload = {"meta": metadata, "metrics": metrics, "results": self.results}
        temporary_path = self.output_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.output_path)

    def run(self) -> Dict[str, Any]:
        data = load_dataset(
            self.data_path,
            self.dataset_config["format"],
            offset=self.args.offset,
            limit=self.limit,
        )
        print("=" * 80)
        print(f"{METHOD_NAME} - {self.dataset}")
        print(
            "Rounds: "
            f"{self.args.n_rounds} | Queries/round: {self.args.n_queries} | "
            f"Top-k/query: {self.args.k_per_query}"
        )
        print(f"Examples: {len(data)} | Output: {self.output_path}")
        print("=" * 80)

        try:
            for index, item in enumerate(tqdm(data, desc=METHOD_NAME), start=1):
                result = self.run_single(item)
                if self.args.omit_retrieval:
                    for round_item in result.get("rounds", []):
                        for query_item in round_item.get("queries", []):
                            query_item.pop("retrieval", None)
                self.results.append(result)

                metrics = compute_metrics(self.results, self.labels)
                flag = "OK" if result["is_correct"] else "NO"
                tqdm.write(
                    f"[{index}/{len(data)}] {flag} "
                    f"GT={result.get('gold', ''):<8} "
                    f"Pred={result.get('predicted', ''):<8} "
                    f"PubMedQ={result.get('num_pubmed_queries', 0)} "
                    f"Docs={result.get('num_retrieved_docs', 0)} "
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
        print(f"Macro-F1 : {metrics['macro_f1']:.4f}")
        print(f"Invalid  : {metrics['invalid_predictions']}")
        print(f"Errors   : {self.error_count}")
        print(f"Saved    : {self.output_path}")
        if self.output_path.suffix.lower() == ".jsonl":
            print(f"Summary  : {self._summary_path()}")
        print("=" * 80)
        return {
            "dataset": self.dataset,
            "metrics": metrics,
            "output_path": str(self.output_path),
            "summary_path": str(self._summary_path()),
            "backbone": self.backbone,
            "method": METHOD_NAME,
        }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="i-MedRAG-PubMed baseline evaluation")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS) + ["all", "PubMedQA", "MedQA", "MedMCQA"],
        default="pubmedqa",
    )
    parser.add_argument("--method", default=METHOD_NAME, choices=[METHOD_NAME])
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-rounds", "--n_rounds", type=int, default=2)
    parser.add_argument("--n-queries", "--n_queries", type=int, default=2)
    parser.add_argument("--k-per-query", "--k_per_query", type=int, default=5)
    parser.add_argument("--max-history-items", "--max_history_items", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--max-abstract-chars", type=int, default=3000)
    parser.add_argument("--max-chars-per-snippet", type=int, default=1200)
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--temperature-query", type=float, default=0.2)
    parser.add_argument("--temperature-answer", type=float, default=0.2)
    parser.add_argument("--temperature-final", type=float, default=0.2)
    parser.add_argument("--max-tokens-query", type=int, default=512)
    parser.add_argument("--max-tokens-answer", type=int, default=512)
    parser.add_argument("--max-tokens-final", type=int, default=512)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--cache-dir", default="cache/medrag_pubmed")
    parser.add_argument("--llm-cache-dir", default="cache/imedrag_pubmed_llm")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-llm-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-deduplicate-pmids", action="store_true")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--omit-retrieval", action="store_true")
    return parser


def print_result_table(run_outputs: List[Dict[str, Any]]) -> None:
    if not run_outputs:
        return

    by_dataset = {item["dataset"]: item for item in run_outputs}
    backbone = run_outputs[0].get("backbone", "")
    method = run_outputs[0].get("method", METHOD_NAME)
    columns = [
        "Backbone",
        "Method",
        "PubMedQA ACC",
        "PubMedQA Macro-F1",
        "MedQA ACC",
        "MedQA Macro-F1",
        "MedMCQA ACC",
        "MedMCQA Macro-F1",
    ]

    def metric(dataset: str, key: str) -> str:
        item = by_dataset.get(dataset)
        if not item:
            return "-"
        value = item["metrics"].get(key, 0.0)
        return f"{value:.6f}"

    row = [
        backbone,
        method,
        metric("pubmedqa", "accuracy"),
        metric("pubmedqa", "macro_f1"),
        metric("medqa", "accuracy"),
        metric("medqa", "macro_f1"),
        metric("medmcqa", "accuracy"),
        metric("medmcqa", "macro_f1"),
    ]
    widths = [max(len(col), len(value)) for col, value in zip(columns, row)]
    print("\nResult table")
    print(" | ".join(col.ljust(width) for col, width in zip(columns, widths)))
    print(" | ".join("-" * width for width in widths))
    print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def run_evaluation(args: argparse.Namespace) -> List[Dict[str, Any]]:
    args.dataset = str(args.dataset).lower()
    if args.save_interval <= 0:
        raise ValueError("--save-interval must be greater than zero")
    if args.dataset == "all" and args.data_path:
        raise ValueError("--data-path can only be used with a single dataset")
    if args.dataset == "all" and args.output:
        raise ValueError("--output can only be used with a single dataset")

    datasets = sorted(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]
    outputs: List[Dict[str, Any]] = []
    for dataset in datasets:
        evaluator = IMedRAGPubMedEvaluator(args, dataset=dataset)
        outputs.append(evaluator.run())
    print_result_table(outputs)
    return outputs


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        run_evaluation(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
