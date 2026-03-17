"""
MedQA 基线测试脚本
测试模型不使用 MEDAR-QA 流程，仅基于自身知识回答选择题
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

from llm_client import get_llm_client


# ==================== 配置参数 ====================
DATA_PATH = "data/medqa_sample.jsonl"          # MedQA 数据集路径
OUTPUT_DIR = "TEST_RESULTS/medqa"              # 结果输出目录
TEST_LIMIT = 500                                 # 测试数量，None 表示全部测试
SAVE_INTERVAL = 50                             # 保存间隔
TEMPERATURE = 0.0                              # 选择题建议尽量低温
MAX_TOKENS = 50                                # 这里只需要输出 A/B/C/D
# ================================================


PROMPT_TEMPLATE = """You are a medical expert answering a multiple-choice medical exam question.

Question:
{question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Instructions:
- Choose the single best answer.
- Respond with ONLY one capital letter: A, B, C, or D.
- Do not provide any explanation.
- Do not output words like "Answer:".

Your response:"""


class MedQABaselineEvaluator:
    """MedQA 基线评估器"""

    def __init__(self):
        self.results = []
        self.correct_count = 0
        self.total_count = 0
        self.parse_fail_count = 0
        self.error_count = 0
        self.llm = get_llm_client()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def load_data(self) -> List[Dict]:
        """加载 MedQA 数据集"""
        data = []
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if TEST_LIMIT is not None and i >= TEST_LIMIT:
                    break
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def extract_answer(self, response: Optional[str]) -> str:
        """
        从模型响应中稳健提取答案选项
        支持：
        - A
        - Answer: A
        - The answer is B
        - (C)
        """
        if not response:
            return ""

        text = response.strip().upper()

        # 1) 最严格：整个响应就是单个字母
        if re.fullmatch(r"[ABCD]", text):
            return text

        # 2) 常见模式优先匹配
        patterns = [
            r"^\s*ANSWER\s*[:：]?\s*([ABCD])\b",
            r"^\s*THE\s+ANSWER\s+IS\s*[:：]?\s*([ABCD])\b",
            r"^\s*OPTION\s*[:：]?\s*([ABCD])\b",
            r"^\s*\(?([ABCD])\)?[\s\.\,:;!]*$",
            r"\b([ABCD])\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return ""

    def build_prompt(self, item: Dict) -> str:
        """构建 prompt"""
        question = item.get("question", "")
        options = item.get("options", {})

        return PROMPT_TEMPLATE.format(
            question=question,
            option_a=options.get("A", ""),
            option_b=options.get("B", ""),
            option_c=options.get("C", ""),
            option_d=options.get("D", ""),
        )

    def run_single_test(self, item: Dict) -> Dict:
        """运行单个测试样本"""
        idx = item.get("realidx", item.get("idx", "unknown"))
        question = item.get("question", "")
        options = item.get("options", {})
        ground_truth = str(item.get("answer_idx", "")).strip().upper()

        prompt = self.build_prompt(item)

        try:
            response = self.llm.chat(
                prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            predicted = self.extract_answer(response)
            is_correct = (predicted == ground_truth)

            print(f"\n[IDX: {idx}] 模型原始响应: {repr(response)}")
            print(f"[IDX: {idx}] 提取答案: {predicted}")

            if predicted == "":
                self.parse_fail_count += 1

            return {
                "idx": idx,
                "question": question,
                "options": options,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "raw_response": response,
                "is_correct": is_correct,
                "error": None,
            }

        except Exception as e:
            self.error_count += 1
            return {
                "idx": idx,
                "question": question,
                "options": options,
                "ground_truth": ground_truth,
                "predicted": "",
                "raw_response": None,
                "is_correct": False,
                "error": str(e),
            }

    def run_evaluation(self):
        """运行完整评估"""
        print("=" * 80)
        print("MedQA 基线测试（纯模型知识）")
        print("=" * 80)

        data = self.load_data()
        print(f"加载了 {len(data)} 条测试数据")

        for i, item in enumerate(tqdm(data, desc="测试进度"), start=1):
            result = self.run_single_test(item)
            self.results.append(result)
            self.total_count += 1

            if result["is_correct"]:
                self.correct_count += 1

            accuracy = (self.correct_count / self.total_count * 100) if self.total_count > 0 else 0.0

            print(f"\n[{i}/{len(data)}] IDX: {result['idx']}")
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
        filename = os.path.join(OUTPUT_DIR, f"baseline_{prefix}{timestamp}.json")
        latest_filename = os.path.join(
            OUTPUT_DIR,
            "baseline_latest.json" if not interim else "baseline_interim_latest.json"
        )

        accuracy = (self.correct_count / self.total_count * 100) if self.total_count > 0 else 0.0

        save_data = {
            "meta": {
                "timestamp": timestamp,
                "test_type": "medqa_baseline",
                "total_count": self.total_count,
                "correct_count": self.correct_count,
                "accuracy": accuracy,
                "test_limit": TEST_LIMIT,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "parse_fail_count": self.parse_fail_count,
                "error_count": self.error_count,
                "data_path": DATA_PATH,
            },
            "results": [
                {
                    "idx": r["idx"],
                    "question": r["question"],
                    "options": r["options"],
                    "ground_truth": r["ground_truth"],
                    "predicted": r["predicted"],
                    "raw_response": r["raw_response"],
                    "is_correct": r["is_correct"],
                    "error": r["error"],
                }
                for r in self.results
            ],
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        with open(latest_filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {filename}")
        print(f"最新结果已更新: {latest_filename}")

    def print_summary(self):
        """打印评估摘要"""
        accuracy = (self.correct_count / self.total_count * 100) if self.total_count > 0 else 0.0

        print("\n" + "=" * 80)
        print("评估摘要 - MedQA 基线测试")
        print("=" * 80)
        print(f"总测试数量: {self.total_count}")
        print(f"正确数量: {self.correct_count}")
        print(f"准确率: {accuracy:.2f}%")
        print(f"解析失败数量: {self.parse_fail_count}")
        print(f"接口错误数量: {self.error_count}")

        # 按标准答案选项统计
        for opt in ["A", "B", "C", "D"]:
            opt_total = sum(1 for r in self.results if str(r["ground_truth"]).upper() == opt)
            opt_correct = sum(
                1 for r in self.results
                if str(r["ground_truth"]).upper() == opt and r["is_correct"]
            )
            if opt_total > 0:
                print(f"  选项 {opt}: {opt_correct}/{opt_total} ({opt_correct / opt_total * 100:.2f}%)")

        print("=" * 80)


def main():
    evaluator = MedQABaselineEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()