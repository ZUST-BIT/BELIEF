"""
MEDAR-QA PubMedQA数据集测试脚本
用于测试PubMedQA数据集在该流程上的准确率
使用简化提示词 Prompt_E_Test_YesNo 进行测试
"""

import json
import os
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

from agents import AgentA, AgentB, AgentC, AgentD, CompletenessController, extract_json_from_response
from retriever import retrieve_process
from llm_client import call_llm
from prompt import Prompt_E_Test_YesNo


# ==================== 配置参数 ====================
DATA_PATH = "data/pubmedqa_hard.json"  # PubMedQA数据集路径
OUTPUT_DIR = "TEST_RESULTS/pubmedqa"               # 结果输出目录
TEST_LIMIT = 10                           # 测试数量，None表示全部测试
MAX_ROUNDS = 1                            # 最大检索轮次
SAVE_INTERVAL = 100                         # 保存间隔
# ================================================


class AgentE_Test_YesNo:
    """简化版AgentE - 用于PubMedQA测试，输出格式为简单JSON"""
    
    def __init__(self):
        self.reasoning_history = []
    
    def add_reasoning_round(self, round_num: int, evidence_count: int, bpa_summary: Dict, note: str):
        from datetime import datetime
        self.reasoning_history.append({
            "round": round_num,
            "evidence_count": evidence_count,
            "bpa_summary": bpa_summary,
            "note": note,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def run(self, question: str, final_decision: Dict, fusion_result: Dict,
            evidence_list: List[Dict]) -> Dict:
        """生成测试答案（与 AgentE.generate_report 保持一致）"""
        # 构建prompt
        prompt = Prompt_E_Test_YesNo.replace("{{QUESTION}}", question)
        prompt = prompt.replace("{{FINAL_DECISION}}", json.dumps(final_decision, indent=2))
        prompt = prompt.replace("{{FUSION_RESULT}}", json.dumps(fusion_result, indent=2))

        # 与真实 AgentE.generate_report 保持一致的证据构建方式
        simplified_evidence = []
        for ev in evidence_list[:10]:
            simplified_evidence.append({
                "source_type": ev.get('source_type', 'Unknown'),
                "metadata": ev.get('metadata', {}),
                "content_snippet": ev.get('content', '')[:500000] + "..."
            })

        prompt = prompt.replace("{{EVIDENCE_LIST}}", json.dumps(simplified_evidence, indent=2, ensure_ascii=False))
        prompt = prompt.replace("{{REASONING_HISTORY}}", json.dumps(self.reasoning_history, indent=2, ensure_ascii=False))

        # 调用LLM
        response = call_llm(prompt, temperature=0, max_tokens=500)

        # 使用与真实 AgentE 相同的健壮 JSON 解析（支持 <think> 块和多种格式）
        result = extract_json_from_response(response)
        if result is not None:
            return result

        # 解析失败时降级：从文本中提取答案
        response_lower = response.lower()
        answer = ""
        if "yes" in response_lower:
            answer = "yes"
        elif "no" in response_lower:
            answer = "no"
        elif "maybe" in response_lower:
            answer = "maybe"
        return {"reasoning": response, "answer": answer}


class PubMedQAEvaluator:
    """PubMedQA数据集评估器"""
    
    def __init__(self):
        self.results = []
        self.correct_count = 0
        self.total_count = 0
        
        # 初始化智能体
        self.agent_a = AgentA()
        self.agent_b = AgentB()
        self.agent_c = AgentC()
        self.agent_d = AgentD()
        self.controller = CompletenessController()
        
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
        
        # 映射各种可能的决策结果
        if decision in ['yes', 'positive', 'support', 'accept']:
            return 'yes'
        elif decision in ['no', 'negative', 'against', 'reject']:
            return 'no'
        elif decision in ['maybe', 'uncertain', 'insufficient_evidence', 'inconclusive']:
            return 'maybe'
        return decision
    
    def extract_decision(self, response: Dict) -> str:
        """从响应中提取decision"""
        # 直接从answer字段获取
        answer = response.get('answer', '')
        if answer and answer.lower() in ['yes', 'no', 'maybe']:
            return answer.lower()
        
        # 从reasoning中尝试提取
        reasoning = response.get('reasoning', '').lower()
        if 'yes' in reasoning:
            return 'yes'
        elif 'no' in reasoning:
            return 'no'
        elif 'maybe' in reasoning:
            return 'maybe'
        
        return ""
    
    def run_single_test(self, item: Dict) -> Dict:
        """运行单个测试样本"""
        pmid = item.get('pmid', 'unknown')
        question = item.get('QUESTION', '')
        contexts = item.get('CONTEXTS', [])
        ground_truth = item.get('final_decision', '')  # 标准答案
        
        # 整理context
        context = self.format_context(contexts)
        
        # 运行推理流程
        try:
            result = self.run_inference(question, context)
            predicted = self.extract_decision(result)
            is_correct = (predicted == self.normalize_decision(ground_truth))
            
            return {
                'pmid': pmid,
                'question': question,
                'context': context,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'reasoning': result.get('reasoning', ''),
                'is_correct': is_correct,
                'full_result': result,
                'error': None
            }
        except Exception as e:
            return {
                'pmid': pmid,
                'question': question,
                'context': context,
                'ground_truth': ground_truth,
                'predicted': None,
                'reasoning': '',
                'is_correct': False,
                'full_result': None,
                'error': str(e)
            }
    
    def run_inference(self, question: str, context: str) -> Dict:
        """执行推理流程（使用简化版AgentE）"""
        agent_e = AgentE_Test_YesNo()  # 每次推理创建新实例
        
        # Step 1: Agent A - 问题分析
        agent_a_result = self.agent_a.run(question)
        fod = agent_a_result.get('frame_of_discernment', ['H', '¬H'])
        
        # 初始化证据列表
        all_evidence = []
        
        # 添加用户提供的上下文作为证据
        if context and context.strip():
            user_context_evidence = {
                "source": "user_provided_context",
                "content": context.strip(),
                "score": 1.0,
                "type": "user_context",
                "metadata": {"is_user_provided": True}
            }
            all_evidence.append(user_context_evidence)
        
        # Step 2: 知识检索
        retrieval_result = retrieve_process(question, agent_a_result)
        all_evidence.extend(retrieval_result)
        
        # Step 3: Agent B - 证据分析
        agent_b_result = self.agent_b.run(question, all_evidence)
        
        # Step 4: Agent C - 证据评估（需要传入 question_pico）
        contextual_question = f"原问题{question}\n当前识别框架为{fod}。"
        question_pico_data = agent_a_result.get('pico_elements', {})
        agent_c_result = self.agent_c.run(
            hypothesis=contextual_question,
            agent_b_result=agent_b_result,
            question_pico=question_pico_data,
            frame_of_discernment=fod,  # 与 main.py 保持一致，传入实际 FoD
            verbose=False
        )
        
        # Step 5: Agent D - 证据融合
        bpa_list = agent_c_result.get('bpa_list', [])
        
        if not bpa_list:
            agent_d_result = {
                "note": "No valid BPA for fusion",
                "final_decision": {
                    "decision": "INSUFFICIENT_EVIDENCE",
                    "confidence": 0.0,
                    "reason": "没有足够的有效证据进行推理"
                },
                "fusion_result": {},
                "belief_plausibility": {}
            }
        else:
            agent_d_result = self.agent_d.run(question, fod, bpa_list)
        
        # Step 6: 构建增强型证据列表（Enhanced Evidence List）
        enhanced_evidence_input = []
        c_evaluations = agent_c_result.get('evaluations', [])
        if c_evaluations:
            for ev in c_evaluations:
                ev_data = ev.get('evaluation', {})
                # 优先使用 Agent C 生成的富文本报告
                rich_content = ev_data.get('content_for_generator')
                
                # 兜底：如果没有富文本，使用原始摘要
                if not rich_content:
                    rich_content = f"[Raw Snippet]: {ev_data.get('processed_input_snippet', 'N/A')}"
                
                enhanced_evidence_input.append({
                    "source": ev.get('source_type', 'Unknown'),
                    "content": rich_content,
                    "score": ev_data.get('bpa_components', {}).get('support_hypothesis', 0),
                    "type": "analyzed_report",
                    "metadata": ev.get('metadata', {})
                })
        else:
            # 回退到原始证据
            enhanced_evidence_input = all_evidence
        
        # Step 7: Agent E - 生成答案（使用简化版）
        final_decision = agent_d_result.get('final_decision', {
            "decision": "UNCERTAIN",
            "confidence": 0.0,
            "reason": "未获得决策结果"
        })
        fusion_result = agent_d_result.get('fusion_result', {})

        agent_e.add_reasoning_round(
            round_num=1,
            evidence_count=len(all_evidence),
            bpa_summary={
                "average_support": sum(b.get('support_hypothesis', 0) for b in bpa_list) / len(bpa_list) if bpa_list else 0,
                "average_against": sum(b.get('against_hypothesis', 0) for b in bpa_list) / len(bpa_list) if bpa_list else 0,
                "average_uncertainty": sum(b.get('uncertainty', 0) for b in bpa_list) / len(bpa_list) if bpa_list else 0,
                "bpa_count": len(bpa_list)
            },
            note="推理完成"
        )
        
        agent_e_result = agent_e.run(
            question=question,
            final_decision=final_decision,
            fusion_result=fusion_result,
            evidence_list=enhanced_evidence_input
        )
        
        return agent_e_result
    
    def run_evaluation(self):
        """运行完整评估"""
        print("=" * 80)
        print("PubMedQA 数据集评估 (简化版AgentE)")
        print("=" * 80)
        
        # 加载数据
        data = self.load_data()
        print(f"加载了 {len(data)} 条测试数据")
        
        # 运行测试
        for i, item in enumerate(tqdm(data, desc="测试进度")):
            result = self.run_single_test(item)
            self.results.append(result)
            self.total_count += 1
            
            if result['is_correct']:
                self.correct_count += 1
            
            # 打印当前进度
            accuracy = self.correct_count / self.total_count * 100
            print(f"\n[{i+1}/{len(data)}] PMID: {result['pmid']}")
            print(f"  标准答案: {result['ground_truth']}, 预测: {result['predicted']}, 正确: {result['is_correct']}")
            # print(f"  推理: {result.get('reasoning', '')[:100]}...")
            print(f"  当前准确率: {accuracy:.2f}%")
            
            # 定期保存
            if (i + 1) % SAVE_INTERVAL == 0:
                self.save_results(interim=True)
        
        # 保存最终结果
        self.save_results(interim=False)
        self.print_summary()
    
    def save_results(self, interim: bool = False):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "interim_" if interim else "final_"
        filename = f"{OUTPUT_DIR}/pubmedqa_{prefix}result_{timestamp}.json"
        
        # 构建保存数据
        save_data = {
            'meta': {
                'timestamp': timestamp,
                'total_count': self.total_count,
                'correct_count': self.correct_count,
                'accuracy': self.correct_count / self.total_count * 100 if self.total_count > 0 else 0,
                'test_limit': TEST_LIMIT,
                'max_rounds': MAX_ROUNDS
            },
            'results': [
                {
                    'pmid': r['pmid'],
                    'question': r['question'],
                    'ground_truth': r['ground_truth'],
                    'predicted': r['predicted'],
                    'reasoning': r.get('reasoning', ''),
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
        print("评估摘要")
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
        
        # 统计错误数量
        error_count = sum(1 for r in self.results if r['error'] is not None)
        if error_count > 0:
            print(f"\n错误数量: {error_count}")
        
        print("=" * 80)


def main():
    evaluator = PubMedQAEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
