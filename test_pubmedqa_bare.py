"""
PubMedQA 基线测试脚本 - 带 Context
测试模型不使用 MEDAR-QA 流程，仅基于自身知识 + 提供的 Context 回答问题
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

from llm_client import get_llm_client


# ==================== 配置参数 ====================
DATA_PATH = "data/pubmedqa_sample2.json"         # PubMedQA 数据集路径
OUTPUT_DIR = "TEST_RESULTS/pubmedqa"            # 结果输出目录
TEST_LIMIT = None                                # 测试数量，None 表示全部测试
SAVE_INTERVAL = 1000                            # 保存间隔

TEMPERATURE = 0.0                               # 分类任务建议低温
MAX_TOKENS = 50                                 # 只需要 yes/no/maybe
MAX_CHARS_PER_CONTEXT = 1200                    # 每段 context 最大字符数
MAX_TOTAL_CONTEXT_CHARS = 5000                  # 整体 context 最大字符数
# ================================================


PROMPT_TEMPLATE = """You are a medical expert.

Based on the provided context, answer the following biomedical yes/no/maybe question.

Context:
{context}

Question:
{question}

Instructions:
- Use the provided context as the primary evidence.
- If the context is insufficient or inconclusive, answer "maybe".
- Respond with ONLY one word: yes, no, or maybe.
- Do not provide any explanation.
- Do not output words like "Answer:".

Your response:"""


class PubMedQABaselineWithContext:
    """PubMedQA 基线评估器（带 Context）"""

    VALID_LABELS = {"yes", "no", "maybe"}

    def __init__(self):
        self.results = []
        self.correct_count = 0
        self.total_count = 0
        self.parse_fail_count = 0
        self.error_count = 0

        self.llm = get_llm_client()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def load_data(self) -> List[Dict]:
        """加载 PubMedQA 数据集"""
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        data = []
        for pmid, item in raw_data.items():
            sample = dict(item)
            sample["pmid"] = pmid
            data.append(sample)

            if TEST_LIMIT is not None and len(data) >= TEST_LIMIT:
                break

        return data

    def truncate_text(self, text: str, max_chars: int) -> str:
        """截断文本"""
        if not text:
            return ""
        text = str(text).strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + " ...[TRUNCATED]"

    def format_context(self, contexts: List[str]) -> str:
        """将 CONTEXTS 列表整理为上下文字符串，并控制长度"""
        if not contexts:
            return "No context provided."

        parts = []
        total_chars = 0

        for i, ctx in enumerate(contexts, start=1):
            ctx = self.truncate_text(ctx, MAX_CHARS_PER_CONTEXT)
            part = f"[{i}] {ctx}"

            if total_chars + len(part) > MAX_TOTAL_CONTEXT_CHARS:
                break

            parts.append(part)
            total_chars += len(part)

        return "\n\n".join(parts) if parts else "No context provided."

    def normalize_decision(self, decision: Optional[str]) -> str:
        """标准化模型输出为 yes / no / maybe"""
        if not decision:
            return ""

        text = decision.strip().lower()

        # 1. 整个输出就是标签
        if text in self.VALID_LABELS:
            return text

        # 2. 常见格式：Answer: yes
        patterns = [
            r"^\s*answer\s*[:：]?\s*(yes|no|maybe)\b",
            r"^\s*the answer is\s*[:：]?\s*(yes|no|maybe)\b",
            r"^\s*(yes|no|maybe)[\s\.\,\;\:\!\?]*$",
            r"\b(yes|no|maybe)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        # 3. 有限兜底
        if "uncertain" in text or "inconclusive" in text or "insufficient" in text:
            return "maybe"

        return ""

    def run_single_test(self, item: Dict) -> Dict:
        """运行单个测试样本"""
        pmid = item.get("pmid", "unknown")
        question = item.get("QUESTION", "")
        contexts = item.get("CONTEXTS", [])
        ground_truth = self.normalize_decision(item.get("final_decision", ""))

        context = self.format_context(contexts)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        try:
            response = self.llm.chat(
                prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            predicted = self.normalize_decision(response)
            parse_failed = (predicted == "")
            if parse_failed:
                self.parse_fail_count += 1

            is_correct = (predicted == ground_truth)

            return {
                "pmid": pmid,
                "question": question,
                "context": context,
                "context_count": len(contexts),
                "used_context_length": len(context),
                "ground_truth": ground_truth,
                "predicted": predicted,
                "raw_response": response,
                "parse_failed": parse_failed,
                "is_correct": is_correct,
                "error": None,
            }

        except Exception as e:
            self.error_count += 1
            return {
                "pmid": pmid,
                "question": question,
                "context": context,
                "context_count": len(contexts),
                "used_context_length": len(context),
                "ground_truth": ground_truth,
                "predicted": "",
                "raw_response": None,
                "parse_failed": False,
                "is_correct": False,
                "error": str(e),
            }

    def run_evaluation(self):
        """运行完整评估"""
        print("=" * 80)
        print("PubMedQA 基线测试（带 Context）")
        print("=" * 80)

        data = self.load_data()
        print(f"加载了 {len(data)} 条测试数据")

        for i, item in enumerate(tqdm(data, desc="测试进度"), start=1):
            result = self.run_single_test(item)
            self.results.append(result)
            self.total_count += 1

            if result["is_correct"]:
                self.correct_count += 1

            accuracy = self.correct_count / self.total_count * 100 if self.total_count > 0 else 0.0

            print(f"\n[{i}/{len(data)}] PMID: {result['pmid']}")
            print(f"  标准答案: {result['ground_truth']}, 预测: {result['predicted']}, 正确: {result['is_correct']}")
            print(f"  当前准确率: {accuracy:.2f}%")

            if i % SAVE_INTERVAL == 0:
                self.save_results(interim=True)

        self.save_results(interim=False)
        self.print_summary()

    def save_results(self, interim: bool = False):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "interim_" if interim else "final_"

        filename = os.path.join(
            OUTPUT_DIR,
            f"pubmedqa_baseline_with_context_{prefix}{timestamp}.json"
        )
        latest_filename = os.path.join(
            OUTPUT_DIR,
            "pubmedqa_baseline_with_context_latest.json"
            if not interim else
            "pubmedqa_baseline_with_context_interim_latest.json"
        )

        accuracy = self.correct_count / self.total_count * 100 if self.total_count > 0 else 0.0
        avg_context_len = (
            sum(r["used_context_length"] for r in self.results) / len(self.results)
            if self.results else 0.0
        )

        save_data = {
            "meta": {
                "timestamp": timestamp,
                "test_type": "baseline_with_context",
                "total_count": self.total_count,
                "correct_count": self.correct_count,
                "accuracy": accuracy,
                "test_limit": TEST_LIMIT,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "max_chars_per_context": MAX_CHARS_PER_CONTEXT,
                "max_total_context_chars": MAX_TOTAL_CONTEXT_CHARS,
                "parse_fail_count": self.parse_fail_count,
                "error_count": self.error_count,
                "avg_context_length": avg_context_len,
                "data_path": DATA_PATH,
            },
            "results": [
                {
                    "pmid": r["pmid"],
                    "question": r["question"],
                    "ground_truth": r["ground_truth"],
                    "predicted": r["predicted"],
                    "raw_response": r["raw_response"],
                    "context_count": r["context_count"],
                    "used_context_length": r["used_context_length"],
                    "parse_failed": r["parse_failed"],
                    "is_correct": r["is_correct"],
                    "error": r["error"],
                }
                for r in self.results
            ]
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        with open(latest_filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {filename}")
        print(f"最新结果已更新: {latest_filename}")

    def print_summary(self):
        """打印评估摘要"""
        accuracy = self.correct_count / self.total_count * 100 if self.total_count > 0 else 0.0

        print("\n" + "=" * 80)
        print("评估摘要 - 基线测试（带 Context）")
        print("=" * 80)
        print(f"总测试数量: {self.total_count}")
        print(f"正确数量: {self.correct_count}")
        print(f"准确率: {accuracy:.2f}%")
        print(f"解析失败数量: {self.parse_fail_count}")
        print(f"接口错误数量: {self.error_count}")

        print("\n按答案类型统计:")
        for label in ["yes", "no", "maybe"]:
            total = sum(1 for r in self.results if r["ground_truth"] == label)
            correct = sum(1 for r in self.results if r["ground_truth"] == label and r["is_correct"])
            if total > 0:
                print(f"  {label.capitalize()}: {correct}/{total} ({correct / total * 100:.2f}%)")

        print("=" * 80)


def main():
    evaluator = PubMedQABaselineWithContext()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()