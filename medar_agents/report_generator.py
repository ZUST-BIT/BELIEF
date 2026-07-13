"""Report generator agent."""

import json
import re
from typing import Dict, Any, List

from prompt import Prompt_E_Test_MCQ, Prompt_E_Test_YesNo
from .json_utils import extract_json_from_response
from .llm_chain import build_llm_chain
from .numeric_utils import clamp_confidence
from .task_modes import SELECTION, normalize_task_mode


class ReportGenerator:
    """
    Generate the final medical QA report for a single reasoning pass.
    """

    def __init__(self):
        self._chain = build_llm_chain(
            lambda prompt: prompt,
            temperature=0.2,
            max_tokens=3000,
        )

    @staticmethod
    def _safe_confidence(value: Any) -> float:
        return clamp_confidence(value)

    @staticmethod
    def _normalize_option_text(value: Any) -> str:
        return re.sub(r"[\W_]+", " ", str(value or "").casefold()).strip()

    @classmethod
    def _extract_mcq_options(cls, question: str) -> Dict[str, str]:
        options: Dict[str, str] = {}
        for pattern in (
            r'"([A-D])"\s*:\s*"([^"]+)"',
            r"'([A-D])'\s*:\s*'([^']+)'",
        ):
            for label, text in re.findall(pattern, question or "", flags=re.IGNORECASE):
                options[label.upper()] = text.strip()

        for pattern in (
            r"^\s*\(([A-D])\)\s*(.+?)\s*$",
            r"^\s*([A-D])\s*[.:)\-]\s*(.+?)\s*$",
        ):
            for label, text in re.findall(
                pattern,
                question or "",
                flags=re.IGNORECASE | re.MULTILINE,
            ):
                options.setdefault(
                    label.upper(),
                    text.strip().rstrip(",").strip().strip("\"'"),
                )
        return options

    @classmethod
    def _normalize_answer(cls, value: Any, is_mcq: bool, question: str = "") -> str:
        answer = str(value or "").strip()
        if is_mcq:
            option = answer.upper()
            label_match = re.fullmatch(r"(?:OPTION\s*)?([A-D])[.)]?", option)
            if label_match:
                return label_match.group(1)

            normalized_answer = cls._normalize_option_text(answer)
            matches = []
            for label, option_text in cls._extract_mcq_options(question).items():
                normalized_option = cls._normalize_option_text(option_text)
                if normalized_answer == normalized_option or (
                    min(len(normalized_answer), len(normalized_option)) >= 4
                    and (
                        normalized_answer in normalized_option
                        or normalized_option in normalized_answer
                    )
                ):
                    matches.append(label)
            return matches[0] if len(set(matches)) == 1 else "UNKNOWN"

        normalized = re.sub(r"[\s\-/]+", "_", answer.lower()).strip("_")
        if normalized in {
            "yes",
            "true",
            "y",
            "1",
            "strong_yes",
            "support_association",
            "fact_confirmed",
            "favor_intervention",
            "h",
        }:
            return "yes"
        if normalized in {
            "no",
            "false",
            "n",
            "0",
            "strong_no",
            "refute_association",
            "fact_contradicted",
            "favor_comparator",
            "no_significant_difference",
            "¬h",
        }:
            return "no"
        return "maybe"

    def _normalize_report(
        self,
        report: Dict[str, Any],
        final_decision: Dict[str, Any],
        is_mcq: bool,
        question: str,
    ) -> Dict[str, Any]:
        normalized = dict(report)
        fallback_answer = final_decision.get("answer", final_decision.get("decision"))
        normalized["answer"] = self._normalize_answer(
            normalized.get("answer", fallback_answer),
            is_mcq=is_mcq,
            question=question,
        )
        normalized["reasoning"] = str(normalized.get("reasoning", "")).strip()
        normalized["confidence_score"] = self._safe_confidence(
            normalized.get(
                "confidence_score",
                final_decision.get("confidence_score", final_decision.get("confidence", 0.0)),
            )
        )
        return normalized

    def generate_report(
        self,
        question: str,
        final_decision: Dict[str, Any],
        fusion_result: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
        task_mode: str = "SELECTION",
    ) -> Dict[str, Any]:
        task_mode = normalize_task_mode(task_mode)
        is_mcq = task_mode == SELECTION

        prompt_template = Prompt_E_Test_MCQ if is_mcq else Prompt_E_Test_YesNo
        prompt = prompt_template.replace("{{QUESTION}}", question)
        prompt = prompt.replace("{{FINAL_DECISION}}", json.dumps(final_decision, indent=2))
        prompt = prompt.replace("{{FUSION_RESULT}}", json.dumps(fusion_result, indent=2))

        simplified_evidence = []
        for ev in evidence_list[:10]:
            simplified_evidence.append({
                "source_type": ev.get("source_type", ev.get("source", "Unknown")),
                "metadata": ev.get("metadata", {}),
                "content_snippet": ev.get("content", "")[:4000],
            })

        prompt = prompt.replace("{{EVIDENCE_LIST}}", json.dumps(simplified_evidence, indent=2, ensure_ascii=False))
        prompt = prompt.replace("{{REASONING_HISTORY}}", "[]")

        response = self._chain.invoke(prompt)
        report = extract_json_from_response(response)
        if isinstance(report, dict):
            return self._normalize_report(
                report,
                final_decision,
                is_mcq=is_mcq,
                question=question,
            )

        print(f"JSON解析失败，原始响应: {response[:500]}...")
        fallback_report = {
            "answer": final_decision.get("answer", final_decision.get("decision")),
            "reasoning": str(response or "")[:500],
            "confidence_score": final_decision.get(
                "confidence_score",
                final_decision.get("confidence", 0.0),
            ),
            "full_report": response,
            "error": "JSON解析失败，返回原始文本",
        }
        return self._normalize_report(
            fallback_report,
            final_decision,
            is_mcq=is_mcq,
            question=question,
        )

    def run(
        self,
        question: str,
        final_decision: Dict[str, Any],
        fusion_result: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
        verbose: bool = False,
        task_mode: str = "SELECTION",
    ) -> Dict[str, Any]:
        if verbose:
            print(f"\n{'='*60}")
            print("[ReportGenerator] 开始生成最终报告")
            print(f"{'='*60}")

        report = self.generate_report(
            question,
            final_decision,
            fusion_result,
            evidence_list,
            task_mode=task_mode,
        )

        if verbose:
            print(f"\n{'='*60}")
            print("[ReportGenerator] 报告生成完成")
            print(f"{'='*60}")
            print(f"\n答案: {report.get('answer', 'N/A')}")
            print(f"置信度: {report.get('confidence_score', 'N/A')}")

            if "full_report" in report:
                print("\n完整报告:")
                print("-" * 60)
                print(report["full_report"])
                print("-" * 60)

        return report
