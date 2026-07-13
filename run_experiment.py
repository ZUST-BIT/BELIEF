"""Unified entry point for MedRAG-PubMed and i-MedRAG-PubMed evaluations."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List


BASELINES = {
    "MedRAG-PubMed": "test_script.medrag_baseline",
    "i-MedRAG-PubMed": "test_script.imedrag_pubmed_baseline",
}

_METHOD_ALIASES = {
    "medrag-pubmed": "MedRAG-PubMed",
    "medrag_pubmed": "MedRAG-PubMed",
    "imedrag-pubmed": "i-MedRAG-PubMed",
    "i-medrag-pubmed": "i-MedRAG-PubMed",
    "i_medrag_pubmed": "i-MedRAG-PubMed",
}

_DATASET_ALIASES = {
    "pubmedqa": "pubmedqa",
    "medqa": "medqa",
    "medmcqa": "medmcqa",
    "all": "all",
}


def _canonical_method(method: str) -> str:
    value = str(method or "").strip()
    if value in BASELINES:
        return value
    canonical = _METHOD_ALIASES.get(value.lower())
    if canonical:
        return canonical
    raise ValueError(
        "unsupported method. Available methods: " + ", ".join(sorted(BASELINES))
    )


def _canonical_dataset(dataset: str) -> str:
    value = str(dataset or "").strip()
    canonical = _DATASET_ALIASES.get(value.lower())
    if canonical:
        return canonical
    raise ValueError("unsupported dataset. Use PubMedQA, MedQA, MedMCQA, or all")


def _preparse(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", default="i-MedRAG-PubMed")
    parser.add_argument("--dataset", default="pubmedqa")
    parser.add_argument("--backbone", default=None)
    known, _ = parser.parse_known_args(argv)
    known.method = _canonical_method(known.method)
    known.dataset = _canonical_dataset(known.dataset)
    return known


def _build_top_level_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a maintained PubMed retrieval baseline evaluation.",
        epilog=(
            "Select a method and add --help to see its full options, for example: "
            "python run_experiment.py --method MedRAG-PubMed --help"
        ),
    )
    parser.add_argument(
        "--method",
        choices=sorted(BASELINES),
        default="i-MedRAG-PubMed",
        help="Baseline implementation to run.",
    )
    parser.add_argument(
        "--dataset",
        choices=["PubMedQA", "MedQA", "MedMCQA", "all"],
        default="PubMedQA",
        help="Evaluation dataset; MedRAG-PubMed does not support 'all'.",
    )
    parser.add_argument(
        "--backbone",
        help="Optional model identifier override (primarily used by i-MedRAG-PubMed).",
    )
    return parser


def _contains_option(argv: List[str], option: str) -> bool:
    return any(token == option or token.startswith(option + "=") for token in argv)


def _set_backbone_env(backbone: str) -> None:
    value = str(backbone or "").strip()
    if not value:
        return
    os.environ["MEDAR_MODEL_NAME"] = value
    os.environ["MEDAR_OLLAMA_MODEL"] = value
    os.environ["MEDAR_VLLM_MODEL"] = value
    os.environ["MEDAR_LOCAL_MODEL_PATH"] = value


def _replace_option(argv: List[str], option: str, value: str) -> List[str]:
    result: List[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == option:
            result.extend([option, value])
            skip_next = True
            continue
        if token.startswith(option + "="):
            result.append(option + "=" + value)
            continue
        result.append(token)
    if option not in result and not any(token.startswith(option + "=") for token in result):
        result.extend([option, value])
    return result


def _remove_options(argv: List[str], option_names: Iterable[str]) -> List[str]:
    names = set(option_names)
    result: List[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        option = token.split("=", 1)[0]
        if option in names:
            if "=" not in token:
                skip_next = True
            continue
        result.append(token)
    return result


def _run_imedrag(argv: List[str]) -> None:
    from test_script.imedrag_pubmed_baseline import build_argument_parser, run_evaluation

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    run_evaluation(args)


def _run_medrag(argv: List[str]) -> None:
    from test_script.medrag_baseline import MedRAGBaselineEvaluator, build_argument_parser

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if str(args.dataset).lower() == "all":
        parser.error("MedRAG-PubMed currently expects one dataset at a time")
    args.dataset = str(args.dataset).lower()
    config = __import__("test_script.medrag_baseline", fromlist=["DATASET_CONFIGS"]).DATASET_CONFIGS[
        args.dataset
    ]
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


def main(argv: List[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if any(token in {"-h", "--help"} for token in raw_argv) and not _contains_option(
        raw_argv, "--method"
    ):
        _build_top_level_parser().print_help()
        return

    try:
        pre = _preparse(raw_argv)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if pre.backbone:
        _set_backbone_env(pre.backbone)

    normalized_argv = _replace_option(raw_argv, "--method", pre.method)
    normalized_argv = _replace_option(normalized_argv, "--dataset", pre.dataset)

    if pre.method == "i-MedRAG-PubMed":
        _run_imedrag(normalized_argv)
        return

    if pre.method == "MedRAG-PubMed":
        medrag_argv = _remove_options(normalized_argv, ["--method", "--backbone"])
        _run_medrag(medrag_argv)
        return

    raise SystemExit(f"unsupported method: {pre.method}")


if __name__ == "__main__":
    main()
