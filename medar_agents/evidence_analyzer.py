"""Evidence analyzer agent."""

from typing import Dict, Any, List, Optional

from prompt import Prompt_B
from .json_utils import extract_json_from_response
from .llm_chain import build_llm_chain


class EvidenceAnalyzer:
    """
    Analyze evidence snippets and extract structured PICO/study info.
    """

    def __init__(self):
        self._chain = build_llm_chain(
            lambda evidence_text: Prompt_B.replace("{{EVIDENCE_TEXT}}", evidence_text),
            temperature=0,
            max_tokens=4096,
        )

    def analyze_evidence(self, evidence_text: str) -> Dict[str, Any]:
        response = self._chain.invoke(evidence_text)
        result = extract_json_from_response(response)
        if result is not None:
            return result
        print(f"JSON解析失败，原始响应: {response[:500]}...")
        return {
            "error": "JSON解析失败",
            "raw_response": response,
        }

    def analyze_evidence_list(self, question: str, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        analyzed_evidences = []

        for idx, evidence in enumerate(evidence_list, 1):
            evidence_content = evidence.get("content", "")
            source_type = evidence.get("source_type", evidence.get("type", "user_input"))

            if evidence_content and len(evidence_content.strip()) > 50:
                analysis = self.analyze_evidence(evidence_content)
                analyzed_evidences.append({
                    "evidence_id": idx,
                    "source_type": source_type,
                    "metadata": evidence.get("metadata", {}),
                    "analysis": analysis,
                    "original_content": (
                        evidence_content[:80000] + "..." if len(evidence_content) > 80000 else evidence_content
                    ),
                })
            else:
                analyzed_evidences.append({
                    "evidence_id": idx,
                    "source_type": source_type,
                    "metadata": evidence.get("metadata", {}),
                    "original_content": (
                        evidence_content[:300] + "..." if len(evidence_content) > 300 else evidence_content
                    ),
                    "analysis": {"note": "Content too short, PICO extraction skipped"},
                })

        return {
            "evidence_count": len(evidence_list),
            "analyzed_evidences": analyzed_evidences,
        }

    def run(self, question: str, evidence_list: Optional[List[Dict[str, Any]]] = None, verbose: bool = False) -> Dict[str, Any]:
        if verbose:
            print(f"\n{'='*60}")
            print(f"[EvidenceAnalyzer] 开始处理问题: {question}")
            print(f"{'='*60}\n")

        if evidence_list is not None:
            result = self.analyze_evidence_list(question, evidence_list)
        else:
            result = {
                "evidence_count": 0,
                "analyzed_evidences": [],
                "error": "evidence_list is required",
            }

        if verbose:
            print(f"\n{'='*60}")
            print("[EvidenceAnalyzer] 处理完成")
            print(f"{'='*60}")
            if "question_analysis" in result:
                print(f"\n问题类型: {result['question_analysis'].get('question_type', 'Unknown')}")
                print(f"分析模式: {result['question_analysis'].get('analysis_mode', 'Unknown')}")
            print(f"证据数量: {result['evidence_count']}")
            print(f"分析的证据数量: {len(result['analyzed_evidences'])}")

            print("\n证据分析摘要（前3个）：")
            for i, ev in enumerate(result["analyzed_evidences"][:3], 1):
                print(f"\n--- 证据 {i} ---")
                print(f"来源: {ev['source_type']}")
                if "analysis" in ev and "pico" in ev["analysis"]:
                    pico = ev["analysis"]["pico"]
                    print(f"P: {pico.get('P', 'N/A')}")
                    print(f"I: {pico.get('I', 'N/A')}")
                    print(f"C: {pico.get('C', 'N/A')}")
                    print(f"O: {pico.get('O', 'N/A')}")
                    print(f"研究类型: {ev['analysis'].get('study_type', 'Unknown')}")

        return result
