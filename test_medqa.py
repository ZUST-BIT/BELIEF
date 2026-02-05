"""
MEDAR-QA MedQA数据集测试脚本
用于测试MedQA数据集在该流程上的准确率
使用简化提示词 Prompt_E_Test_MCQ 进行测试
"""

import json
import re
import os
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

from agents import AgentA, AgentB, AgentC, AgentD, CompletenessController
from retriever import retrieve_process
from llm_client import call_llm
from prompt import Prompt_E_Test_MCQ


def extract_json_from_response(response: str) -> dict:
    """
    从LLM响应中提取JSON对象（健壮版本）
    """
    if not response:
        return None
    
    # 方法1: 尝试提取 ```json ... ``` 代码块
    if "```json" in response:
        try:
            json_str = response.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass
    
    # 方法2: 尝试提取 ``` ... ``` 代码块
    if "```" in response:
        try:
            json_str = response.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass
    
    # 方法3: 使用正则表达式查找最后一个完整的JSON对象
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    if matches:
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # 方法4: 查找从第一个 { 到最后一个 } 的内容
    first_brace = response.find('{')
    last_brace = response.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            json_str = response[first_brace:last_brace + 1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # 方法5: 直接尝试解析整个响应
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    return None


# ==================== 配置参数 ====================
DATA_PATH = "data/medqa_sample.jsonl"  # MedQA数据集路径
OUTPUT_DIR = "TEST_RESULTS/medqa"       # 结果输出目录
TEST_LIMIT = None                         # 测试数量，None表示全部测试
MAX_ROUNDS = 1                          # 最大检索轮次
SAVE_INTERVAL = 100                       # 保存间隔
# ================================================


class AgentE_Test:
    """简化版AgentE - 用于测试，输出格式为简单JSON"""
    
    def __init__(self):
        self.reasoning_history = []
    
    def add_reasoning_round(self, round_num: int, evidence_count: int, bpa_summary: Dict, note: str):
        self.reasoning_history.append({
            "round": round_num,
            "evidence_count": evidence_count,
            "bpa_summary": bpa_summary,
            "note": note
        })
    
    def run(self, question: str, final_decision: Dict, fusion_result: Dict, 
            belief_analysis: Dict, evidence_list: List[Dict]) -> Dict:
        """生成测试答案"""
        # 构建prompt
        prompt = Prompt_E_Test_MCQ.replace("{{QUESTION}}", question)
        prompt = prompt.replace("{{FINAL_DECISION}}", json.dumps(final_decision, indent=2))
        prompt = prompt.replace("{{FUSION_RESULT}}", json.dumps(fusion_result, indent=2))
        prompt = prompt.replace("{{BELIEF_ANALYSIS}}", json.dumps(belief_analysis, indent=2))
        
        # 简化证据列表
        simplified_evidence = []
        for ev in evidence_list[:10]:
            simplified_evidence.append({
                "source": ev.get('source_type', ev.get('source', 'Unknown')),
                "content": ev.get('content', '')[:2000]
            })
        
        prompt = prompt.replace("{{EVIDENCE_LIST}}", json.dumps(simplified_evidence, indent=2, ensure_ascii=False))
        prompt = prompt.replace("{{REASONING_HISTORY}}", json.dumps(self.reasoning_history, indent=2, ensure_ascii=False))
        
        # 调用LLM
        response = call_llm(prompt, temperature=0, max_tokens=500)
        
        # 解析JSON响应（使用健壮版本）
        result = extract_json_from_response(response)
        if result is not None:
            return result
        else:
            # 尝试直接提取答案字母
            answer = ""
            for char in response.upper():
                if char in ['A', 'B', 'C', 'D']:
                    answer = char
                    break
            return {"reasoning": response, "answer": answer}


class MedQAEvaluator:
    """MedQA数据集评估器"""
    
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
        """加载MedQA数据集"""
        data = []
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if TEST_LIMIT and i >= TEST_LIMIT:
                    break
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
    
    def format_question(self, item: Dict) -> str:
        """格式化问题和选项"""
        question = item['question']
        options = item['options']
        
        formatted = f'"question": "{question}",\n"options": {{\n'
        for key, value in options.items():
            formatted += f'    "{key}": "{value}",\n'
        return formatted.rstrip(',\n') + '\n}'
    
    def extract_answer(self, response: Dict) -> str:
        """从响应中提取答案选项"""
        # 直接从answer字段获取
        answer = response.get('answer', '')
        if answer and answer.upper() in ['A', 'B', 'C', 'D']:
            return answer.upper()
        
        # 从reasoning中尝试提取
        reasoning = response.get('reasoning', '')
        for char in reasoning.upper():
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        return ""
    
    def run_single_question(self, item: Dict) -> Dict:
        """对单个问题运行MEDAR-QA流程"""
        question = self.format_question(item)
        agent_e = AgentE_Test()  # 使用简化版AgentE
        
        # Agent A: 问题分析
        agent_a_result = self.agent_a.run(question)
        fod = agent_a_result.get('frame_of_discernment', ['H', '¬H'])
        question_pico_data = agent_a_result.get('pico_elements', {})
        
        # 检索与推理循环
        all_evidence = []
        current_round = 1
        
        while current_round <= MAX_ROUNDS:
            # 知识检索
            retrieval_result = retrieve_process(question, agent_a_result)
            all_evidence.extend(retrieval_result)
            
            # Agent B: 证据分析
            agent_b_result = self.agent_b.run(question, all_evidence)
            
            # Agent C: 证据评估（添加 question_pico 参数）
            contextual_question = f"原问题{question}\n当前识别框架为{fod}。"
            agent_c_result = self.agent_c.run(
                hypothesis=contextual_question,
                agent_b_result=agent_b_result,
                question_pico=question_pico_data,
                verbose=False
            )
            
            # Agent D: 证据融合
            bpa_list = agent_c_result.get('bpa_list', [])
            
            if not bpa_list:
                agent_d_result = {
                    "final_decision": {"decision": "INSUFFICIENT_EVIDENCE", "confidence": 0.0, "reason": "证据不足"}
                }
            else:
                agent_d_result = self.agent_d.run(question, fod, bpa_list)
            
            # 记录推理轮次
            bpa_summary = {
                "bpa_count": len(bpa_list),
                "avg_support": sum(b.get('support_hypothesis', 0) for b in bpa_list) / len(bpa_list) if bpa_list else 0
            }
            agent_e.add_reasoning_round(current_round, len(retrieval_result), bpa_summary, f"第{current_round}轮")
            
            # 完备性检查
            fused_bpa = agent_d_result.get('fusion_result', {}).get('fused_bpa', {'uncertainty': 1.0})
            belief_pl = agent_d_result.get('belief_plausibility', {})
            conflict = agent_d_result.get('fusion_result', {}).get('conflict_coefficient', 0)
            
            completeness = self.controller.analyze_completeness(fused_bpa, belief_pl, conflict)
            if not completeness['should_continue'] or current_round >= MAX_ROUNDS:
                break
            current_round += 1
        
        # 构建增强型证据列表（Enhanced Evidence List）
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
        
        # Agent E: 生成答案（使用简化版和增强型证据）
        agent_e_result = agent_e.run(
            question=question,
            final_decision=agent_d_result.get('final_decision', {}),
            fusion_result=agent_d_result.get('fusion_result', {}),
            belief_analysis=agent_d_result.get('belief_plausibility', {}),
            evidence_list=enhanced_evidence_input
        )
        
        return {
            "question": question,
            "agent_d_fusion": agent_d_result,
            "final_report": agent_e_result
        }
    
    def run(self):
        """运行评估"""
        data = self.load_data()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        category_stats = {}
        
        print("\n" + "="*70)
        print("MEDAR-QA MedQA数据集评估")
        print("="*70)
        print(f"数据集: {DATA_PATH}")
        print(f"测试数量: {len(data)}")
        print(f"最大轮次: {MAX_ROUNDS}")
        print("="*70 + "\n")
        
        for i, item in enumerate(tqdm(data, desc="评估进度")):
            try:
                print(f"\n[{i+1}/{len(data)}] 问题 {item.get('realidx', i)}")
                
                response = self.run_single_question(item)
                final_report = response.get('final_report', {})
                predicted = self.extract_answer(final_report)
                ground_truth = item['answer_idx']
                is_correct = predicted.upper() == ground_truth.upper() if predicted else False
                
                if is_correct:
                    self.correct_count += 1
                self.total_count += 1
                
                # 分类统计
                meta = item.get('meta_info', 'unknown')
                if meta not in category_stats:
                    category_stats[meta] = {'correct': 0, 'total': 0}
                category_stats[meta]['total'] += 1
                if is_correct:
                    category_stats[meta]['correct'] += 1
                
                self.results.append({
                    "realidx": item.get('realidx', i),
                    "question": item['question'],
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "reasoning": final_report.get('reasoning', ''),
                    "is_correct": is_correct,
                    "meta_info": meta
                })
                
                print(f"正确: {ground_truth} | 预测: {predicted} | {'✓' if is_correct else '✗'}")
                print(f"推理: {final_report.get('reasoning', '')[:100]}...")
                print(f"准确率: {self.correct_count}/{self.total_count} = {self.correct_count/self.total_count*100:.1f}%")
                
                # 定期保存
                if (i + 1) % SAVE_INTERVAL == 0:
                    self._save_checkpoint(timestamp)
                    
            except Exception as e:
                print(f"错误: {e}")
                self.results.append({
                    "realidx": item.get('realidx', i),
                    "error": str(e),
                    "is_correct": False
                })
                self.total_count += 1
        
        # 保存最终结果
        self._save_final(timestamp, category_stats)
    
    def _save_checkpoint(self, timestamp: str):
        """保存检查点"""
        acc = self.correct_count / self.total_count if self.total_count > 0 else 0
        output = os.path.join(OUTPUT_DIR, f"medqa_{timestamp}_checkpoint.json")
        with open(output, 'w', encoding='utf-8') as f:
            json.dump({"accuracy": acc, "results": self.results}, f, ensure_ascii=False, indent=2)
        print(f"[检查点已保存]")
    
    def _save_final(self, timestamp: str, category_stats: dict):
        """保存最终结果"""
        accuracy = self.correct_count / self.total_count if self.total_count > 0 else 0
        
        summary = {
            "timestamp": timestamp,
            "total": self.total_count,
            "correct": self.correct_count,
            "accuracy": f"{accuracy*100:.2f}%",
            "category_stats": {k: {**v, 'accuracy': f"{v['correct']/v['total']*100:.1f}%"} 
                              for k, v in category_stats.items()},
            "results": self.results
        }
        
        output = os.path.join(OUTPUT_DIR, f"medqa_{timestamp}.json")
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print("评估完成")
        print("="*70)
        print(f"总题数: {self.total_count}")
        print(f"正确数: {self.correct_count}")
        print(f"准确率: {accuracy*100:.2f}%")
        print("\n分类准确率:")
        for cat, stats in category_stats.items():
            acc = stats['correct']/stats['total']*100 if stats['total'] > 0 else 0
            print(f"  {cat}: {stats['correct']}/{stats['total']} = {acc:.1f}%")
        print(f"\n结果保存: {output}")
        print("="*70)


if __name__ == "__main__":
    evaluator = MedQAEvaluator()
    evaluator.run()
