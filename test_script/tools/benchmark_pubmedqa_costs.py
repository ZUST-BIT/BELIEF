"""
Benchmark token cost and time efficiency on PubMedQA for the maintained
MedRAG-PubMed and i-MedRAG-PubMed baselines.

Outputs:
- Detailed per-question records (JSON)
- Per-method summary (JSON)

Example:
    python benchmark_pubmedqa_costs.py --limit 50
    python benchmark_pubmedqa_costs.py --output-dir TEST_RESULTS/cost_benchmark
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import llm_client


DATA_PATH = "datasets/pubmedqa.json"
FALLBACK_DATA_PATH = "data/test_pubmedqa.json"
DEFAULT_OUTPUT_DIR = "TEST_RESULTS/cost_benchmark"
DEFAULT_METHODS = (
    "medrag_pubmed",
    "imedrag_pubmed",
)
ALL_METHODS = DEFAULT_METHODS
METHOD_ALIASES = {
    "medrag": "medrag_pubmed",
    "medrag-pubmed": "medrag_pubmed",
    "imedrag": "imedrag_pubmed",
    "i-medrag": "imedrag_pubmed",
    "i-medrag-pubmed": "imedrag_pubmed",
    "i_medrag_pubmed": "imedrag_pubmed",
}
PUBMEDQA_OPTIONS = {"yes": "yes", "no": "no", "maybe": "maybe"}


def normalize_method_names(raw_methods: str) -> List[str]:
    raw = str(raw_methods or "").strip()
    if not raw or raw.lower() == "all":
        return list(ALL_METHODS)

    normalized_methods: List[str] = []
    for item in raw.split(","):
        method = item.strip().lower()
        if not method:
            continue
        method = METHOD_ALIASES.get(method, method)
        if method not in ALL_METHODS:
            valid = ", ".join(ALL_METHODS)
            raise ValueError(f"Unknown method '{item.strip()}'. Valid methods: {valid}")
        if method not in normalized_methods:
            normalized_methods.append(method)

    if not normalized_methods:
        raise ValueError("--methods did not contain any valid method names")
    return normalized_methods


def normalize_yesno(text: Optional[str]) -> str:
    if not text:
        return ""
    s = str(text)
    s = re.sub(r"<think>.*?</think>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in s:
        s = s.split("</think>")[-1]
    s = s.strip().lower()

    if s in {"yes", "no", "maybe"}:
        return s

    patterns = [
        r"^\s*answer\s*[:：]?\s*(yes|no|maybe)\b",
        r"^\s*the answer is\s*[:：]?\s*(yes|no|maybe)\b",
        r"\b(yes|no|maybe)\b",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(1)

    if any(w in s for w in ["uncertain", "inconclusive", "insufficient"]):
        return "maybe"

    return ""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


class LLMCallRecorder:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def snapshot(self) -> int:
        return len(self.records)

    def record(
        self,
        backend: str,
        elapsed_s: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated: bool,
        ok: bool,
    ) -> None:
        self.records.append(
            {
                "backend": backend,
                "elapsed_s": elapsed_s,
                "prompt_tokens": int(max(0, prompt_tokens)),
                "completion_tokens": int(max(0, completion_tokens)),
                "total_tokens": int(max(0, total_tokens)),
                "estimated": bool(estimated),
                "ok": bool(ok),
            }
        )

    def aggregate_since(self, start_idx: int) -> Dict[str, Any]:
        subset = self.records[start_idx:]
        return {
            "llm_calls": len(subset),
            "llm_calls_ok": sum(1 for r in subset if r["ok"]),
            "prompt_tokens": sum(r["prompt_tokens"] for r in subset),
            "completion_tokens": sum(r["completion_tokens"] for r in subset),
            "total_tokens": sum(r["total_tokens"] for r in subset),
            "estimated_calls": sum(1 for r in subset if r["estimated"]),
            "llm_elapsed_s": sum(r["elapsed_s"] for r in subset),
        }


def install_llm_instrumentation(recorder: LLMCallRecorder) -> None:
    # OpenAI-compatible backend
    def openai_chat_instrumented(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        start = time.perf_counter()
        prompt_toks = 0
        completion_toks = 0
        total_toks = 0
        estimated = False
        ok = False
        content = ""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = self._build_payload(prompt, temperature, max_tokens)
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=1200,
            )
            response.raise_for_status()
            result = response.json()

            usage = result.get("usage", {}) if isinstance(result, dict) else {}
            prompt_toks = int(usage.get("prompt_tokens", 0) or 0)
            completion_toks = int(usage.get("completion_tokens", 0) or 0)
            total_toks = int(usage.get("total_tokens", 0) or 0)

            raw = result["choices"][0]["message"]["content"].strip()
            content = llm_client._remove_think_tags(raw)
            ok = True
            return content
        finally:
            elapsed = time.perf_counter() - start
            if total_toks <= 0:
                estimated = True
                prompt_toks = estimate_tokens(prompt)
                completion_toks = estimate_tokens(content)
                total_toks = prompt_toks + completion_toks
            recorder.record(
                backend="openai_compatible",
                elapsed_s=elapsed,
                prompt_tokens=prompt_toks,
                completion_tokens=completion_toks,
                total_tokens=total_toks,
                estimated=estimated,
                ok=ok,
            )

    # Ollama backend
    def ollama_chat_instrumented(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        start = time.perf_counter()
        prompt_toks = 0
        completion_toks = 0
        total_toks = 0
        estimated = False
        ok = False
        content = ""

        try:
            disable_thinking = llm_client.get_disable_thinking_for_model(self.model)
            system_prompt = llm_client._get_model_system_prompt(self.model)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if system_prompt:
                payload["system"] = system_prompt
            if disable_thinking:
                payload["think"] = False

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=1200,
            )
            response.raise_for_status()
            result = response.json()

            prompt_toks = int(result.get("prompt_eval_count", 0) or 0)
            completion_toks = int(result.get("eval_count", 0) or 0)
            total_toks = prompt_toks + completion_toks

            raw = (result.get("response", "") or "").strip()
            content = llm_client._remove_think_tags(raw)
            ok = True
            return content
        finally:
            elapsed = time.perf_counter() - start
            if total_toks <= 0:
                estimated = True
                prompt_toks = estimate_tokens(prompt)
                completion_toks = estimate_tokens(content)
                total_toks = prompt_toks + completion_toks
            recorder.record(
                backend="ollama",
                elapsed_s=elapsed,
                prompt_tokens=prompt_toks,
                completion_tokens=completion_toks,
                total_tokens=total_toks,
                estimated=estimated,
                ok=ok,
            )

    # Transformers backend
    original_tf_chat = llm_client.TransformersClient.chat

    def transformers_chat_instrumented(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        start = time.perf_counter()
        ok = False
        content = ""
        try:
            content = original_tf_chat(self, prompt, temperature, max_tokens)
            ok = True
            return content
        finally:
            elapsed = time.perf_counter() - start
            estimated = False
            try:
                if getattr(self, "_tokenizer", None) is not None:
                    prompt_toks = len(self._tokenizer(prompt, return_tensors="pt")["input_ids"][0])
                    completion_toks = len(self._tokenizer(content, return_tensors="pt")["input_ids"][0])
                else:
                    estimated = True
                    prompt_toks = estimate_tokens(prompt)
                    completion_toks = estimate_tokens(content)
            except Exception:
                estimated = True
                prompt_toks = estimate_tokens(prompt)
                completion_toks = estimate_tokens(content)

            recorder.record(
                backend="transformers",
                elapsed_s=elapsed,
                prompt_tokens=prompt_toks,
                completion_tokens=completion_toks,
                total_tokens=prompt_toks + completion_toks,
                estimated=estimated,
                ok=ok,
            )

    llm_client.OpenAICompatibleClient.chat = openai_chat_instrumented
    llm_client.OllamaClient.chat = ollama_chat_instrumented
    llm_client.TransformersClient.chat = transformers_chat_instrumented


class TimedPubMedRetriever:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.reset()

    def reset(self) -> None:
        self.call_count = 0
        self.total_latency_s = 0.0
        self.total_docs = 0
        self.unique_ids = set()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        requested_k = top_k if top_k is not None else k
        old_top_k = getattr(self.inner, "top_k", None)
        changed_top_k = requested_k is not None and old_top_k is not None
        if changed_top_k:
            self.inner.top_k = int(requested_k)

        started = time.perf_counter()
        results: List[Dict[str, Any]] = []
        try:
            results = list(self.inner.retrieve(query) or [])
            return results
        finally:
            if changed_top_k:
                self.inner.top_k = old_top_k
            self.call_count += 1
            self.total_latency_s += time.perf_counter() - started
            self.total_docs += len(results)
            for item in results:
                doc_id = str(item.get("id") or item.get("pmid") or "").strip()
                if doc_id:
                    self.unique_ids.add(doc_id)

    def stats(self) -> Dict[str, Any]:
        return {
            "num_pubmed_queries": self.call_count,
            "num_retrieved_docs": self.total_docs,
            "num_unique_retrieved_docs": len(self.unique_ids),
            "pubmed_retrieval_elapsed_s": round(self.total_latency_s, 4),
        }


class MedRAGPubMedRunner:
    def __init__(self) -> None:
        from medrag import MedRAGPipeline, PubMedBM25Retriever

        self.llm = llm_client.get_llm_client()
        self.retriever = TimedPubMedRetriever(
            PubMedBM25Retriever(
                top_k=5,
                candidate_k=50,
                max_abstract_chars=3000,
                cache_dir="cache/medrag_pubmed",
                request_timeout=30,
                retry=2,
            )
        )
        self.pipeline = MedRAGPipeline(
            retriever=self.retriever,
            llm=self.llm,
            temperature=0.0,
            max_tokens=500,
            max_chars_per_snippet=1800,
            max_context_chars=10000,
        )
        self.last_extra: Dict[str, Any] = {}

    def predict(self, question: str, contexts: List[str]) -> str:
        self.retriever.reset()
        output: Optional[Dict[str, Any]] = None
        try:
            output = self.pipeline.run(
                question=question,
                options=PUBMEDQA_OPTIONS,
                sample_id="cost_benchmark",
                dataset_name="pubmedqa",
            )
            return normalize_yesno(output.get("answer", ""))
        finally:
            self.last_extra = self.retriever.stats()
            if output is not None:
                retrieval = output.get("retrieval", []) or []
                self.last_extra.update({
                    "num_retrieved_docs": len(retrieval),
                })


class IMedRAGPubMedRunner:
    def __init__(self) -> None:
        from medrag import DEFAULT_I_MEDRAG_PUBMED_CONFIG, PubMedBM25Retriever, run_i_medrag_pubmed

        self.llm = llm_client.get_llm_client()
        self.retriever = TimedPubMedRetriever(
            PubMedBM25Retriever(
                top_k=5,
                candidate_k=50,
                max_abstract_chars=3000,
                cache_dir="cache/medrag_pubmed",
                request_timeout=30,
                retry=2,
            )
        )
        self.run_i_medrag_pubmed = run_i_medrag_pubmed
        self.config = dict(DEFAULT_I_MEDRAG_PUBMED_CONFIG)
        self.config.update(
            {
                "n_rounds": 2,
                "n_queries": 2,
                "k_per_query": 5,
                "cache_llm_outputs": False,
                "cache_retrieval": True,
                "retrieval_cache_dir": "cache/medrag_pubmed",
                "max_tokens_query": 512,
                "max_tokens_answer": 512,
                "max_tokens_final": 512,
            }
        )
        self.last_extra: Dict[str, Any] = {}

    def predict(self, question: str, contexts: List[str]) -> str:
        self.retriever.reset()
        output: Optional[Dict[str, Any]] = None
        try:
            output = self.run_i_medrag_pubmed(
                question=question,
                options=PUBMEDQA_OPTIONS,
                dataset_name="pubmedqa",
                llm=self.llm,
                pubmed_retriever=self.retriever,
                config=self.config,
            )
            return normalize_yesno(
                output.get("final_prediction", "") or output.get("answer", "")
            )
        finally:
            self.last_extra = self.retriever.stats()
            if output is not None:
                self.last_extra.update({
                    "num_pubmed_queries": output.get("num_pubmed_queries", 0),
                    "num_retrieved_docs": output.get("num_retrieved_docs", 0),
                    "num_unique_retrieved_docs": output.get("num_unique_retrieved_docs", 0),
                    "num_followup_rounds": output.get("num_followup_rounds", 0),
                    "num_followup_queries": output.get("num_followup_queries", 0),
                    "method_reported_input_tokens": output.get("input_tokens", 0),
                    "method_reported_output_tokens": output.get("output_tokens", 0),
                    "method_reported_latency_s": round(float(output.get("latency_seconds", 0.0) or 0.0), 4),
                })


@dataclass
class MethodExecutionResult:
    prediction: str
    elapsed_s: float
    token_stats: Dict[str, Any]
    error: Optional[str]
    extra: Dict[str, Any]


class PubMedQACostBenchmark:
    def __init__(
        self,
        limit: Optional[int],
        output_dir: str,
        data_path: Optional[str],
        method_names: Optional[List[str]] = None,
    ) -> None:
        self.limit = limit
        self.output_dir = output_dir
        self.data_path = self._resolve_data_path(data_path)
        raw_method_names = method_names or list(DEFAULT_METHODS)
        self.method_names = [METHOD_ALIASES.get(name, name) for name in raw_method_names]
        os.makedirs(self.output_dir, exist_ok=True)

        self.recorder = LLMCallRecorder()
        install_llm_instrumentation(self.recorder)

        self.data = self._load_data()
        self.methods = self._build_methods()

    @staticmethod
    def _resolve_data_path(data_path: Optional[str]) -> str:
        candidates = [data_path] if data_path else [DATA_PATH, FALLBACK_DATA_PATH]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        checked = ", ".join(str(candidate) for candidate in candidates if candidate)
        raise FileNotFoundError(f"PubMedQA data file not found. Checked: {checked}")

    def _load_data(self) -> List[Dict[str, Any]]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items: List[Dict[str, Any]] = []
        for pmid, item in raw.items():
            sample = dict(item)
            sample["pmid"] = pmid
            items.append(sample)
            if self.limit is not None and len(items) >= self.limit:
                break
        return items

    def _build_methods(self) -> Dict[str, Any]:
        builders = {
            "medrag_pubmed": lambda: MedRAGPubMedRunner(),
            "imedrag_pubmed": lambda: IMedRAGPubMedRunner(),
        }
        unknown = [name for name in self.method_names if name not in builders]
        if unknown:
            valid = ", ".join(builders)
            raise ValueError(f"unknown method(s): {', '.join(unknown)}. Valid: {valid}")

        methods: Dict[str, Any] = {}
        for method_name in self.method_names:
            try:
                methods[method_name] = builders[method_name]()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to initialize required method '{method_name}'"
                ) from exc
        return methods

    def _run_one_method(self, method_name: str, runner: Any, question: str, contexts: List[str]) -> MethodExecutionResult:
        start_idx = self.recorder.snapshot()
        t0 = time.perf_counter()
        prediction = ""
        error = None

        try:
            prediction = runner.predict(question, contexts)
        except Exception as e:
            error = str(e)

        elapsed = time.perf_counter() - t0
        token_stats = self.recorder.aggregate_since(start_idx)
        extra = dict(getattr(runner, "last_extra", {}) or {})

        return MethodExecutionResult(
            prediction=prediction,
            elapsed_s=elapsed,
            token_stats=token_stats,
            error=error,
            extra=extra,
        )

    @staticmethod
    def _safe_percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        s = sorted(values)
        k = (len(s) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        d0 = s[f] * (c - k)
        d1 = s[c] * (k - f)
        return d0 + d1

    def _summarize_method(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        elapsed = [r["elapsed_s"] for r in rows if r["error"] is None]
        total_tokens = [r["token_stats"]["total_tokens"] for r in rows]
        prompt_tokens = [r["token_stats"]["prompt_tokens"] for r in rows]
        completion_tokens = [r["token_stats"]["completion_tokens"] for r in rows]
        llm_calls = [r["token_stats"]["llm_calls"] for r in rows]
        pubmed_queries = [
            int(r.get("extra", {}).get("num_pubmed_queries", 0) or 0) for r in rows
        ]
        retrieved_docs = [
            int(r.get("extra", {}).get("num_retrieved_docs", 0) or 0) for r in rows
        ]

        attempted = len(rows)
        succeeded = sum(1 for r in rows if r["error"] is None)
        errors = attempted - succeeded

        valid_pred_rows = [r for r in rows if r.get("prediction") in {"yes", "no", "maybe"}]
        correct = sum(1 for r in valid_pred_rows if r.get("is_correct") is True)
        accuracy = (correct / attempted) if attempted else 0.0

        return {
            "attempted": attempted,
            "succeeded": succeeded,
            "errors": errors,
            "valid_predictions": len(valid_pred_rows),
            "invalid_predictions": attempted - len(valid_pred_rows),
            "correct": correct,
            "accuracy": round(accuracy, 6),
            "total_elapsed_s": round(sum(elapsed), 4),
            "avg_elapsed_s": round((statistics.mean(elapsed) if elapsed else 0.0), 4),
            "p50_elapsed_s": round(self._safe_percentile(elapsed, 0.5), 4),
            "p95_elapsed_s": round(self._safe_percentile(elapsed, 0.95), 4),
            "total_prompt_tokens": int(sum(prompt_tokens)),
            "total_completion_tokens": int(sum(completion_tokens)),
            "total_tokens": int(sum(total_tokens)),
            "avg_total_tokens_per_q": round((statistics.mean(total_tokens) if total_tokens else 0.0), 2),
            "avg_prompt_tokens_per_q": round((statistics.mean(prompt_tokens) if prompt_tokens else 0.0), 2),
            "avg_completion_tokens_per_q": round((statistics.mean(completion_tokens) if completion_tokens else 0.0), 2),
            "total_llm_calls": int(sum(llm_calls)),
            "avg_llm_calls_per_q": round((statistics.mean(llm_calls) if llm_calls else 0.0), 3),
            "estimated_token_calls": int(sum(r["token_stats"]["estimated_calls"] for r in rows)),
            "total_pubmed_queries": int(sum(pubmed_queries)),
            "avg_pubmed_queries_per_q": round((statistics.mean(pubmed_queries) if pubmed_queries else 0.0), 3),
            "total_retrieved_docs": int(sum(retrieved_docs)),
            "avg_retrieved_docs_per_q": round((statistics.mean(retrieved_docs) if retrieved_docs else 0.0), 3),
        }

    def run(self) -> Tuple[str, str]:
        print("=" * 90)
        print("PubMedQA cost benchmark")
        print(f"Samples: {len(self.data)}")
        print(f"Methods: {', '.join(self.methods.keys())}")
        print(f"LLM backend: {llm_client.LLM_BACKEND}, model: {llm_client.MODEL_NAME if hasattr(llm_client, 'MODEL_NAME') else 'N/A'}")
        print("=" * 90)

        per_method_rows: Dict[str, List[Dict[str, Any]]] = {k: [] for k in self.methods.keys()}
        detail_records: List[Dict[str, Any]] = []

        for item in tqdm(self.data, desc="Benchmark progress"):
            pmid = item.get("pmid", "unknown")
            question = item.get("QUESTION", "")
            contexts = item.get("CONTEXTS", [])
            gt = normalize_yesno(item.get("final_decision", ""))

            sample_detail: Dict[str, Any] = {
                "pmid": pmid,
                "ground_truth": gt,
                "question": question,
                "methods": {},
            }

            for method_name, runner in self.methods.items():
                result = self._run_one_method(method_name, runner, question, contexts)
                pred = normalize_yesno(result.prediction)
                is_correct = (pred == gt) if pred else False

                row = {
                    "pmid": pmid,
                    "ground_truth": gt,
                    "prediction": pred,
                    "is_correct": is_correct,
                    "elapsed_s": round(result.elapsed_s, 4),
                    "token_stats": result.token_stats,
                    "extra": result.extra,
                    "error": result.error,
                }
                per_method_rows[method_name].append(row)
                sample_detail["methods"][method_name] = row

            detail_records.append(sample_detail)

        summary = {
            "meta": {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "data_path": self.data_path,
                "limit": self.limit,
                "sample_count": len(self.data),
                "requested_methods": self.method_names,
                "methods": list(self.methods.keys()),
                "llm_backend": llm_client.LLM_BACKEND,
                "model_name": getattr(llm_client, "MODEL_NAME", None),
            },
            "summary": {m: self._summarize_method(rows) for m, rows in per_method_rows.items()},
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        detail_path = os.path.join(self.output_dir, f"pubmedqa_cost_detail_{ts}.json")
        summary_path = os.path.join(self.output_dir, f"pubmedqa_cost_summary_{ts}.json")

        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump({"meta": summary["meta"], "results": detail_records}, f, ensure_ascii=False, indent=2)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\nSaved files:")
        print(f"- {detail_path}")
        print(f"- {summary_path}")

        print("\nMethod summary:")
        for method_name, stats in summary["summary"].items():
            print(
                f"[{method_name}] "
                f"acc={stats['accuracy']:.4f}, "
                f"avg_time={stats['avg_elapsed_s']:.3f}s, "
                f"p95={stats['p95_elapsed_s']:.3f}s, "
                f"avg_tokens={stats['avg_total_tokens_per_q']:.1f}, "
                f"total_tokens={stats['total_tokens']}, "
                f"avg_pubmed_q={stats['avg_pubmed_queries_per_q']:.2f}, "
                f"avg_docs={stats['avg_retrieved_docs_per_q']:.2f}, "
                f"errors={stats['errors']}"
            )

        return detail_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark token cost and time on PubMedQA")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N samples")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=f"PubMedQA JSON file with PMID keys (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(ALL_METHODS),
        help=(
            "Comma-separated method names, or 'all'. Valid names: "
            + ", ".join(ALL_METHODS)
            + ". Aliases include medrag and imedrag."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save benchmark outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method_names = normalize_method_names(args.methods)
    benchmark = PubMedQACostBenchmark(
        limit=args.limit,
        output_dir=args.output_dir,
        data_path=args.data_path,
        method_names=method_names,
    )
    benchmark.run()


if __name__ == "__main__":
    main()
