"""Direct LLM reasoning agent."""

import json
import re
from typing import Dict, Any, Optional

from config import set_argument
from prompt import Prompt_DirectLLM, Prompt_DirectLLM_YesNo
from .json_utils import extract_json_from_response
from .llm_chain import build_llm_chain


class DirectReasoningAgent:
    """
    Direct LLM reasoning branch, bypassing DS fusion.
    """

    def __init__(self):
        self.args = set_argument()
        self._chain = build_llm_chain(
            lambda prompt: prompt,
            temperature=0.4,
            max_tokens=4096,
        )

    def _repair_common_json_issues(self, response: str) -> Optional[Dict[str, Any]]:
        if not response:
            return None

        repaired = response.strip()

        repaired = re.sub(r"^```json\s*", "", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"^```\s*", "", repaired)
        repaired = re.sub(r"\s*```\s*$", "", repaired)

        repaired = re.sub(
            r'("key_evidence_used"\s*:\s*\[\s*)"([A-Za-z0-9_\-]+)"\s*:\s*([^\]\n\r]+?)(\s*\])',
            r'\1{"\2": \3}\4',
            repaired,
            flags=re.DOTALL,
        )

        first_brace = repaired.find("{")
        last_brace = repaired.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            repaired = repaired[first_brace:last_brace + 1]

        try:
            return json.loads(repaired, strict=False)
        except Exception:
            return None

    def _normalize_direct_llm_result(self, result: Dict[str, Any], is_mcq: bool) -> Dict[str, Any]:
        if not isinstance(result, dict):
            result = {}

        conf = result.get("confidence_score", 0.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        result["confidence_score"] = max(0.0, min(1.0, conf))

        if is_mcq:
            sel = str(result.get("selected_option", "UNKNOWN")).strip().upper()
            result["selected_option"] = sel if sel in {"A", "B", "C", "D"} else "UNKNOWN"
            result.setdefault("key_evidence_used", [])
            return result

        ans = str(result.get("answer", "UNKNOWN")).strip().lower()
        tendency = str(result.get("directional_tendency", "")).strip().lower()

        if ans in {"yes", "no", "maybe"}:
            normalized_answer = ans
        elif ans in {"true", "y", "1"}:
            normalized_answer = "yes"
        elif ans in {"false", "n", "0"}:
            normalized_answer = "no"
        elif ans in {"uncertain", "unknown", "inconclusive"}:
            normalized_answer = "maybe"
        else:
            normalized_answer = "maybe" if tendency in {"lean_yes", "lean_no", "balanced"} else "UNKNOWN"

        result["answer"] = normalized_answer
        result["directional_tendency"] = tendency if tendency in {"lean_yes", "lean_no", "balanced"} else "balanced"
        result.setdefault("uncertainty_note", "")
        result.setdefault("key_evidence_used", [])
        return result

    def run(
        self,
        question: str,
        evidence_result: Dict[str, Any],
        task_mode: str = "SELECTION",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        is_mcq = (task_mode == "SELECTION")

        if verbose:
            print(f"\n{'='*60}")
            mode_label = "MCQ（选择题）" if is_mcq else "Yes/No（是非题）"
            print(f"[DirectReasoningAgent] 开始直接推理 | 题型: {mode_label}")
            print(f"{'='*60}")

        analyzed_evidences = evidence_result.get("analyzed_evidences", [])
        evidence_summaries = []
        for ev in analyzed_evidences[:15]:
            summary: Dict[str, Any] = {
                "evidence_id": ev.get("evidence_id"),
                "source_type": ev.get("source_type", "Unknown"),
                "metadata": ev.get("metadata", {}),
            }
            analysis = ev.get("analysis", {})
            if analysis:
                summary["clinical_summary"] = analysis.get("clinical_summary", "")
                summary["study_design"] = analysis.get("study_design", analysis.get("study_type", ""))
                pico = analysis.get("pico", {})
                if pico:
                    summary["pico"] = pico
            if not summary.get("clinical_summary"):
                summary["content_snippet"] = ev.get("original_content", "")[:600]
            evidence_summaries.append(summary)

        prompt_template = Prompt_DirectLLM
        prompt = prompt_template.replace("{{QUESTION}}", question)
        prompt = prompt.replace(
            "{{ANALYZED_EVIDENCE}}",
            json.dumps(evidence_summaries, ensure_ascii=False, indent=2),
        )

        response = self._chain.invoke(prompt)
        result = extract_json_from_response(response)

        if result is None:
            result = self._repair_common_json_issues(response)

        if result is None:
            print(f"[DirectReasoningAgent] JSON解析失败，原始响应: {response[:300]}...")
            result = {
                "selected_option" if is_mcq else "answer": "UNKNOWN",
                "reasoning": response[:500],
                "confidence_score": 0.0,
                "key_evidence_used": [],
                "error": "JSON解析失败",
            }

        result = self._normalize_direct_llm_result(result, is_mcq=is_mcq)

        if verbose:
            ans_key = "selected_option" if is_mcq else "answer"
            print(f"[DirectReasoningAgent] 答案: {result.get(ans_key)}")
            print(f"[DirectReasoningAgent] 置信度:   {result.get('confidence_score')}")

        return result
