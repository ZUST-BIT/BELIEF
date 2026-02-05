"""
PubMedQA 基线测试脚本 - 带Context
测试模型不使用MEDAR-QA流程，仅基于自身知识+提供的Context回答问题
"""

import json
import os
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

from llm_client import get_llm_client


# ==================== 配置参数 ====================
DATA_PATH = "data/pubmedqa_sample.json"  # PubMedQA数据集路径
OUTPUT_DIR = "TEST_RESULTS/pubmedqa"               # 结果输出目录
TEST_LIMIT = 5                           # 测试数量，None表示全部测试
SAVE_INTERVAL = 1000                         # 保存间隔
# ================================================

# 提示词模板
PROMPT_TEMPLATE = """You are a medical expert. Based on the given context, answer the following yes/no question.

Context:
{context}

Question: {question}

Please analyze the context carefully and answer with ONLY one of the following: "yes", "no", or "maybe".
Your answer should be based on whether the context supports a positive answer to the question.

Answer (yes/no/maybe):"""


class PubMedQABaselineWithContext:
    """PubMedQA基线评估器（带Context）"""
    
    def __init__(self):
        self.results = []
        self.correct_count = 0
        self.total_count = 0
        self.llm = get_llm_client()
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def load_data(self) -> List[Dict]:
        """加载PubMedQA数据集"""
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        data = []
        for pmid, item in raw_data.items():
            item['pmid'] = pmid
            data.append(item)
            if TEST_LIMIT and len(data) >= TEST_LIMIT:
                break
        return data
    
    def format_context(self, contexts: List[str]) -> str:
        """将CONTEXTS列表整理为上下文字符串"""
        return "\n".join(contexts)
    
    def normalize_decision(self, decision: str) -> str:
        """标准化decision为yes/no/maybe"""
        if not decision:
            return ""
        decision = decision.lower().strip()
        
        # 提取第一个有效答案
        for word in decision.split():
            word = word.strip('.,;:!?"\'')
            if word in ['yes', 'no', 'maybe']:
                return word
        
        # 备用匹配
        if 'yes' in decision:
            return 'yes'
        elif 'no' in decision:
            return 'no'
        elif 'maybe' in decision or 'uncertain' in decision:
            return 'maybe'
        return decision
    
    def run_single_test(self, item: Dict) -> Dict:
        """运行单个测试样本"""
        pmid = item.get('pmid', 'unknown')
        question = item.get('QUESTION', '')
        contexts = item.get('CONTEXTS', [])
        ground_truth = item.get('final_decision', '')
        
        context = self.format_context(contexts)
        # context = ""
        # 构建prompt
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        
        try:
            # 调用LLM
            response = self.llm.chat(prompt, temperature=0.1, max_tokens=50000)
            # print(f"PMID {pmid} 的模型响应是：{response}")
            predicted = self.normalize_decision(response)
            is_correct = (predicted == self.normalize_decision(ground_truth))
            
            return {
                'pmid': pmid,
                'question': question,
                'context': context,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'raw_response': response,
                'is_correct': is_correct,
                'error': None
            }
        except Exception as e:
            return {
                'pmid': pmid,
                'question': question,
                'context': context,
                'ground_truth': ground_truth,
                'predicted': None,
                'raw_response': None,
                'is_correct': False,
                'error': str(e)
            }
    
    def run_evaluation(self):
        """运行完整评估"""
        print("=" * 80)
        print("PubMedQA 基线测试（带Context）")
        print("=" * 80)
        
        data = self.load_data()
        print(f"加载了 {len(data)} 条测试数据")
        
        for i, item in enumerate(tqdm(data, desc="测试进度")):
            result = self.run_single_test(item)
            self.results.append(result)
            self.total_count += 1
            
            if result['is_correct']:
                self.correct_count += 1
            
            accuracy = self.correct_count / self.total_count * 100
            print(f"\n[{i+1}/{len(data)}] PMID: {result['pmid']}")
            print(f"  标准答案: {result['ground_truth']}, 预测: {result['predicted']}, 正确: {result['is_correct']}")
            print(f"  当前准确率: {accuracy:.2f}%")
            
            if (i + 1) % SAVE_INTERVAL == 0:
                self.save_results(interim=True)
        
        self.save_results(interim=False)
        self.print_summary()
    
    def save_results(self, interim: bool = False):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "interim_" if interim else "final_"
        filename = f"{OUTPUT_DIR}/pubmedqa_baseline_no_context_{prefix}{timestamp}.json"
        
        save_data = {
            'meta': {
                'timestamp': timestamp,
                'test_type': 'baseline_with_context',
                'total_count': self.total_count,
                'correct_count': self.correct_count,
                'accuracy': self.correct_count / self.total_count * 100 if self.total_count > 0 else 0,
                'test_limit': TEST_LIMIT
            },
            'results': [
                {
                    'pmid': r['pmid'],
                    'question': r['question'],
                    'ground_truth': r['ground_truth'],
                    'predicted': r['predicted'],
                    'raw_response': r['raw_response'],
                    'is_correct': r['is_correct'],
                    'error': r['error']
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {filename}")
    
    def print_summary(self):
        """打印评估摘要"""
        print("\n" + "=" * 80)
        print("评估摘要 - 基线测试（带Context）")
        print("=" * 80)
        print(f"总测试数量: {self.total_count}")
        print(f"正确数量: {self.correct_count}")
        print(f"准确率: {self.correct_count / self.total_count * 100:.2f}%")
        
        # 按答案类型统计
        yes_correct = sum(1 for r in self.results if r['ground_truth'] == 'yes' and r['is_correct'])
        yes_total = sum(1 for r in self.results if r['ground_truth'] == 'yes')
        no_correct = sum(1 for r in self.results if r['ground_truth'] == 'no' and r['is_correct'])
        no_total = sum(1 for r in self.results if r['ground_truth'] == 'no')
        maybe_correct = sum(1 for r in self.results if r['ground_truth'] == 'maybe' and r['is_correct'])
        maybe_total = sum(1 for r in self.results if r['ground_truth'] == 'maybe')
        
        print(f"\n按答案类型统计:")
        if yes_total > 0:
            print(f"  Yes: {yes_correct}/{yes_total} ({yes_correct/yes_total*100:.2f}%)")
        if no_total > 0:
            print(f"  No: {no_correct}/{no_total} ({no_correct/no_total*100:.2f}%)")
        if maybe_total > 0:
            print(f"  Maybe: {maybe_correct}/{maybe_total} ({maybe_correct/maybe_total*100:.2f}%)")
        
        error_count = sum(1 for r in self.results if r['error'] is not None)
        if error_count > 0:
            print(f"\n错误数量: {error_count}")
        
        print("=" * 80)


def main():
    evaluator = PubMedQABaselineWithContext()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
