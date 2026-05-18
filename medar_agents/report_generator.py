"""Report generator agent."""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from config import set_argument
from prompt import Prompt_E_Test_MCQ, Prompt_E_Test_YesNo
from .json_utils import extract_json_from_response
from .llm_chain import build_llm_chain


class ReportGenerator:
    """
    Generate final medical QA report and track reasoning history.
    """

    def __init__(self):
        self.args = set_argument()
        self.reasoning_history = []
        self._chain = build_llm_chain(
            lambda prompt: prompt,
            temperature=0.2,
            max_tokens=3000,
        )

    def add_reasoning_round(self, round_num: int, evidence_count: int, bpa_summary: Dict[str, Any], note: str):
        self.reasoning_history.append({
            "round": round_num,
            "evidence_count": evidence_count,
            "bpa_summary": bpa_summary,
            "note": note,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    def generate_report(
        self,
        question: str,
        final_decision: Dict[str, Any],
        fusion_result: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
        reasoning_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if reasoning_history is None:
            reasoning_history = self.reasoning_history

        prompt = Prompt_E_Test_MCQ.replace("{{QUESTION}}", question)
        prompt = prompt.replace("{{FINAL_DECISION}}", json.dumps(final_decision, indent=2))
        prompt = prompt.replace("{{FUSION_RESULT}}", json.dumps(fusion_result, indent=2))

        simplified_evidence = []
        for ev in evidence_list[:10]:
            simplified_evidence.append({
                "source_type": ev.get("source_type", "Unknown"),
                "metadata": ev.get("metadata", {}),
                "content_snippet": ev.get("content", "")[:500000] + "...",
            })

        prompt = prompt.replace("{{EVIDENCE_LIST}}", json.dumps(simplified_evidence, indent=2, ensure_ascii=False))
        prompt = prompt.replace("{{REASONING_HISTORY}}", json.dumps(reasoning_history, indent=2, ensure_ascii=False))

        response = self._chain.invoke(prompt)
        report = extract_json_from_response(response)
        if report is not None:
            return report
        print(f"JSON解析失败，原始响应: {response[:500]}...")
        return {
            "direct_answer": f"基于当前证据，{final_decision['decision']}",
            "decision": final_decision["decision"],
            "confidence_level": "moderate",
            "full_report": response,
            "error": "JSON解析失败，返回原始文本",
        }

    def run(
        self,
        question: str,
        final_decision: Dict[str, Any],
        fusion_result: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
        verbose: bool = False,
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
            self.reasoning_history,
        )

        if verbose:
            print(f"\n{'='*60}")
            print("[ReportGenerator] 报告生成完成")
            print(f"{'='*60}")
            print(f"\n直接答案: {report.get('direct_answer', 'N/A')}")
            print(f"决策: {report.get('decision', 'N/A')}")
            print(f"置信水平: {report.get('confidence_level', 'N/A')}")

            if "full_report" in report:
                print("\n完整报告:")
                print("-" * 60)
                print(report["full_report"])
                print("-" * 60)

        return report
