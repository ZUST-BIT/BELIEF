"""
PubMedQA RAG测试脚本
测试模型使用检索增强（RAG），但不使用完整MEDAR-QA流程
简化版：检索 + Context合并 + 直接生成
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
DATA_PATH = "data/pubmedqa_sample.json"  # PubMedQA数据集路径
OUTPUT_DIR = "TEST_RESULTS"               # 结果输出目录
TEST_LIMIT = 100                           # 测试数量，None表示全部测试
SAVE_INTERVAL = 1000                         # 保存间隔
# ================================================

# 提示词模板（带检索内容）
PROMPT_TEMPLATE = """You are a medical expert. Based on the provided context, answer the following yes/no question.

Context (Retrieved Evidence + Provided Context):
{context}

Question: {question}

Please analyze the context carefully and answer with ONLY one of the following: "yes", "no", or "maybe".
Your answer should be based on whether the context supports a positive answer to the question.

Answer (yes/no/maybe):"""


class PubMedQARAGEvaluator:
    """PubMedQA RAG评估器（检索增强生成）"""
    
    def __init__(self):
        self.results = []
        self.correct_count = 0
        self.total_count = 0
        self.llm = get_llm_client()
        self.agent_a = AgentA()  # 用于实体提取，辅助检索
        
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
    
    def format_context(self, evidence_list: List[Dict], original_context: str) -> str:
        """将检索到的证据和原始Context合并格式化为上下文字符串"""
        context_parts = []
        
        # 1. 添加原始Context（来自数据集）
        if original_context and original_context.strip():
            # 限制原始上下文长度，避免过长
            original_trimmed = original_context.strip()
            if len(original_trimmed) > 2000:
                original_trimmed = original_trimmed[:2000] + "..."
            context_parts.append(f"[Original Context]:\n{original_trimmed}")
        
        # 2. 添加检索到的证据
        if evidence_list:
            context_parts.append("\n[Retrieved Evidence]:")
            for i, evidence in enumerate(evidence_list[:5], 1):  # 最多取5条证据
                content = evidence.get('content', '')
                source = evidence.get('source', evidence.get('source_type', 'unknown'))
                if content:
                    # 限制单条证据长度
                    content_trimmed = content[:100000] if len(content) > 100000 else content
                    context_parts.append(f"[{i}] ({source}): {content_trimmed}")
        
        combined = "\n\n".join(context_parts) if context_parts else "No relevant evidence found."
        
        # 最终总长度保护（避免超过模型上下文窗口）
        if len(combined) > 6000:
            combined = combined[:6000] + "\n...[Context truncated due to length]"
        
        return combined
    
    def normalize_decision(self, decision: str) -> str:
        """标准化decision为yes/no/maybe"""
        if not decision:
            return ""
        
        # 处理Qwen3的think标签 - 只取最终回答部分
        if '</think>' in decision:
            decision = decision.split('</think>')[-1].strip()
        
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
        
        # 合并原始Context
        original_context = "\n".join(contexts) if contexts else ""
        
        try:
            # Step 1: 使用Agent A进行实体提取
            agent_a_result = self.agent_a.run(question)
            
            # Step 2: 执行检索
            retrieval_result = retrieve_process(question, agent_a_result)
            
            # Step 3: 合并原始Context和检索内容
            combined_context = self.format_context(retrieval_result, original_context)
            
            # Step 4: 构建prompt并调用LLM
            prompt = PROMPT_TEMPLATE.format(
                context=combined_context,
                question=question
            )
            
            response = self.llm.chat(prompt, temperature=0.1, max_tokens=500)
            
            # 如果响应为空，尝试重试一次（可能是模型问题）
            if not response or not response.strip():
                print(f"⚠️ PMID {pmid} 第一次响应为空，重试中...")
                response = self.llm.chat(prompt, temperature=0.2, max_tokens=500)
            
            print(f"PMID {pmid} 的模型响应是：{response[:200] if response else '(空)'}")
            
            # 如果还是空，记录详细信息用于调试
            if not response or not response.strip():
                print(f"❌ PMID {pmid} 响应依然为空！")
                print(f"   Question: {question[:100]}...")
                print(f"   Context length: {len(combined_context)}")
            
            predicted = self.normalize_decision(response)
            is_correct = (predicted == self.normalize_decision(ground_truth))
            
            return {
                'pmid': pmid,
                'question': question,
                'original_context': original_context,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'raw_response': response,
                'retrieved_count': len(retrieval_result),
                'combined_context': combined_context,
                'is_correct': is_correct,
                'error': None
            }
        except Exception as e:
            return {
                'pmid': pmid,
                'question': question,
                'original_context': original_context,
                'ground_truth': ground_truth,
                'predicted': None,
                'raw_response': None,
                'retrieved_count': 0,
                'combined_context': None,
                'is_correct': False,
                'error': str(e)
            }
    
    def run_evaluation(self):
        """运行完整评估"""
        print("=" * 80)
        print("PubMedQA RAG测试（检索增强生成 + Context）")
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
        filename = f"{OUTPUT_DIR}/pubmedqa_rag_{prefix}{timestamp}.json"
        
        save_data = {
            'meta': {
                'timestamp': timestamp,
                'test_type': 'pubmedqa_rag',
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
        print("评估摘要 - PubMedQA RAG测试")
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
        
        # 检索统计
        avg_retrieved = sum(r['retrieved_count'] for r in self.results) / len(self.results) if self.results else 0
        print(f"\n平均检索证据数: {avg_retrieved:.1f}")
        
        error_count = sum(1 for r in self.results if r['error'] is not None)
        if error_count > 0:
            print(f"\n错误数量: {error_count}")
        
        print("=" * 80)


def main():
    evaluator = PubMedQARAGEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
