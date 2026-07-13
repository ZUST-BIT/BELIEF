"""Question analyzer agent."""

import json
from typing import Dict, Any

from prompt import Prompt_A
from .json_utils import extract_json_from_response
from .llm_chain import build_llm_chain


class QuestionAnalyzer:
    """
    Analyze the biomedical question and return a structured JSON config.
    """

    def __init__(self):
        self._chain = build_llm_chain(
            lambda question: Prompt_A.replace("{{QUESTION}}", question),
            temperature=0,
            max_tokens=4096,
        )

    def analyze_question(self, question: str) -> Dict[str, Any]:
        response = self._chain.invoke(question)
        result = extract_json_from_response(response)
        if result is not None:
            return result
        print(f"JSON解析失败，原始响应: {response[:500]}...")
        return {
            "error": "JSON解析失败",
            "raw_response": response,
        }

    def run(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        if verbose:
            print(f"[QuestionAnalyzer] 正在分析问题: {question}")

        result = self.analyze_question(question)

        if verbose:
            print("[QuestionAnalyzer] 分析完成")
            print(f"结果: {json.dumps(result, indent=2, ensure_ascii=False)}")

        return result
