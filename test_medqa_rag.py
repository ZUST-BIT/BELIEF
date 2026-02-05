"""
MedQA RAG基线测试脚本
测试模型使用检索增强（RAG），但不使用完整MEDAR-QA流程
简化版：检索 + 直接生成
"""

import json
import os
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

from llm_client import get_llm_client
from agents import AgentA
from retriever import retrieve_process


# ==================== 配置参数 ====================
DATA_PATH = "data/medqa_sample.jsonl"   # MedQA数据集路径
OUTPUT_DIR = "TEST_RESULTS/medqa"              # 结果输出目录
TEST_LIMIT = None                          # 测试数量，None表示全部测试
SAVE_INTERVAL = 100                        # 保存间隔
# ================================================

# 提示词模板（带检索内容）
PROMPT_TEMPLATE = """You are a medical expert. Answer the following multiple-choice question based on the provided context.

Context (Retrieved Evidence):
{context}

Question: {question}

Options:
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}

Based on the context and your medical knowledge, please analyze the question carefully and respond with ONLY the letter of the correct answer (A, B, C, or D).

Answer:"""


class MedQARAGEvaluator:
    """MedQA RAG评估器（检索增强生成）"""
    
    def __init__(self):
        self.results = []
        self.correct_count = 0
        self.total_count = 0
        self.llm = get_llm_client()
        self.agent_a = AgentA()  # 用于实体提取，辅助检索
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def load_data(self) -> List[Dict]:
        """加载MedQA数据集"""
        data = []
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if TEST_LIMIT and i >= TEST_LIMIT:
                    break
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
    
    def format_context(self, evidence_list: List[Dict]) -> str:
        """将检索到的证据格式化为上下文字符串"""
        if not evidence_list:
            return "No relevant evidence found."
        
        context_parts = []
        for i, evidence in enumerate(evidence_list[:5], 1):  # 最多取5条证据
            content = evidence.get('content', '')
            source = evidence.get('source', 'unknown')
            if content:
                context_parts.append(f"[{i}] ({source}): {content}")
        
        return "\n\n".join(context_parts) if context_parts else "No relevant evidence found."
    
    def extract_answer(self, response: str) -> str:
        """从响应中提取答案选项"""
        if not response:
            return ""
        response = response.strip().upper()
        
        # 直接匹配单个字母
        if response in ['A', 'B', 'C', 'D']:
            return response
        
        # 提取第一个有效选项
        for char in response:
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        return ""
    
    def run_single_test(self, item: Dict) -> Dict:
        """运行单个测试样本"""
        idx = item.get('realidx', 'unknown')
        question = item.get('question', '')
        options = item.get('options', {})
        ground_truth = item.get('answer_idx', '')
        
        try:
            # Step 1: 使用Agent A进行实体提取
            agent_a_result = self.agent_a.run(question)
            
            # Step 2: 执行检索
            retrieval_result = retrieve_process(question, agent_a_result)
            
            # Step 3: 格式化检索内容
            context = self.format_context(retrieval_result)
            
            # Step 4: 构建prompt并调用LLM
            prompt = PROMPT_TEMPLATE.format(
                context=context,
                question=question,
                option_a=options.get('A', ''),
                option_b=options.get('B', ''),
                option_c=options.get('C', ''),
                option_d=options.get('D', '')
            )
            
            response = self.llm.chat(prompt, temperature=0.1, max_tokens=50)
            predicted = self.extract_answer(response)
            is_correct = (predicted == ground_truth.upper())
            
            return {
                'idx': idx,
                'question': question,
                'options': options,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'raw_response': response,
                'retrieved_count': len(retrieval_result),
                'context': context,
                'is_correct': is_correct,
                'error': None
            }
        except Exception as e:
            return {
                'idx': idx,
                'question': question,
                'options': options,
                'ground_truth': ground_truth,
                'predicted': None,
                'raw_response': None,
                'retrieved_count': 0,
                'context': None,
                'is_correct': False,
                'error': str(e)
            }
    
    def run_evaluation(self):
        """运行完整评估"""
        print("=" * 80)
        print("MedQA RAG测试（检索增强生成）")
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
            print(f"\n[{i+1}/{len(data)}] IDX: {result['idx']}")
            print(f"  检索到: {result['retrieved_count']} 条证据")
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
        filename = f"{OUTPUT_DIR}/medqa_rag_{prefix}{timestamp}.json"
        
        save_data = {
            'meta': {
                'timestamp': timestamp,
                'test_type': 'medqa_rag',
                'total_count': self.total_count,
                'correct_count': self.correct_count,
                'accuracy': self.correct_count / self.total_count * 100 if self.total_count > 0 else 0,
                'test_limit': TEST_LIMIT
            },
            'results': [
                {
                    'idx': r['idx'],
                    'question': r['question'],
                    'ground_truth': r['ground_truth'],
                    'predicted': r['predicted'],
                    'raw_response': r['raw_response'],
                    'retrieved_count': r['retrieved_count'],
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
        print("评估摘要 - MedQA RAG测试")
        print("=" * 80)
        print(f"总测试数量: {self.total_count}")
        print(f"正确数量: {self.correct_count}")
        print(f"准确率: {self.correct_count / self.total_count * 100:.2f}%")
        
        # 平均检索数量
        avg_retrieved = sum(r['retrieved_count'] for r in self.results) / len(self.results) if self.results else 0
        print(f"平均检索证据数: {avg_retrieved:.2f}")
        
        # 按选项统计
        print(f"\n按选项统计:")
        for opt in ['A', 'B', 'C', 'D']:
            opt_correct = sum(1 for r in self.results if r['ground_truth'].upper() == opt and r['is_correct'])
            opt_total = sum(1 for r in self.results if r['ground_truth'].upper() == opt)
            if opt_total > 0:
                print(f"  选项{opt}: {opt_correct}/{opt_total} ({opt_correct/opt_total*100:.2f}%)")
        
        error_count = sum(1 for r in self.results if r['error'] is not None)
        if error_count > 0:
            print(f"\n错误数量: {error_count}")
        
        print("=" * 80)


def main():
    evaluator = MedQARAGEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
