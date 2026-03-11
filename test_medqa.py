"""
MEDAR-QA MedQA 数据集批量测试脚本
--------------------------------------
流程与 main.py 完全一致：
    A  检索  B  C  D  完备性控制  E  答案提取
支持多选题（MCQ）批量评估准确率
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from tqdm import tqdm

from agents import AgentA, AgentB, AgentC, AgentD, AgentE, CompletenessController
from retriever import retrieve_process


# ========================= 配置区 =========================
DATA_PATH    = "data/medqa_sample.jsonl"   # MedQA 数据集路径
OUTPUT_DIR   = "TEST_RESULTS/medqa"        # 结果输出目录
TEST_LIMIT   = None                        # 测试题目数量上限，None = 全量
MAX_ROUNDS   = 1                           # 最大检索轮次（与 main.py 保持一致）
SAVE_INTERVAL = 100                          # 每隔多少题自动保存检查点
# =========================================================


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def extract_json_from_response(response: str) -> Optional[dict]:
    """从 LLM 响应中健壮地提取 JSON 对象"""
    if not response:
        return None

    # 1. ```json ... ```
    if "```json" in response:
        try:
            return json.loads(response.split("```json")[1].split("```")[0].strip())
        except (IndexError, json.JSONDecodeError):
            pass

    # 2. ``` ... ```
    if "```" in response:
        try:
            return json.loads(response.split("```")[1].split("```")[0].strip())
        except (IndexError, json.JSONDecodeError):
            pass

    # 3. 正则匹配最后一个顶层 JSON 对象
    for match in reversed(re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # 4. 从第一个 { 到最后一个 }
    fb, lb = response.find('{'), response.rfind('}')
    if fb != -1 and lb > fb:
        try:
            return json.loads(response[fb:lb + 1])
        except json.JSONDecodeError:
            pass

    # 5. 整体解析
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    return None


def format_mcq_question(item: Dict) -> str:
    """
    将 MedQA 条目格式化为与 main.py 一致的问题字符串
    （同时包含 question 正文与 options）
    """
    question = item["question"]
    options  = item["options"]
    opts_str = "\n".join(f'    "{k}": "{v}"' for k, v in options.items())
    return (
        f'"question": "{question}",\n'
        f'"options": {{\n{opts_str}\n}}'
    )


def extract_predicted_answer(final_report: Dict) -> str:
    """从 AgentE 最终报告中提取预测的选项字母"""
    # 优先从 selected_option 字段
    answer = final_report.get("selected_option", "")
    if answer and str(answer).upper() in "ABCD" and len(str(answer).strip()) == 1:
        return answer.upper()

    # 从 direct_answer 字段
    direct = final_report.get("direct_answer", "")
    for ch in str(direct).upper():
        if ch in "ABCD":
            return ch

    # 从 reasoning 字段
    reasoning = final_report.get("reasoning", "")
    for ch in str(reasoning).upper():
        if ch in "ABCD":
            return ch

    # 从 full_report 字段
    full = final_report.get("full_report", "")
    for ch in str(full).upper():
        if ch in "ABCD":
            return ch

    return ""


# ------------------------------------------------------------------
# 核心评估器
# ------------------------------------------------------------------

class MedQAEvaluator:
    """MedQA 数据集批量评估器（流程与 main.py 完全对齐）"""

    def __init__(self):
        # 初始化全部智能体（与 main.py 一致）
        self.agent_a    = AgentA()
        self.agent_b    = AgentB()
        self.agent_c    = AgentC()
        self.agent_d    = AgentD()
        self.controller = CompletenessController()

        self.results       : List[Dict] = []
        self.correct_count : int        = 0
        self.total_count   : int        = 0

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    #  数据加载 
    def load_data(self) -> List[Dict]:
        data = []
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if TEST_LIMIT is not None and i >= TEST_LIMIT:
                    break
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    #  单题推理（完整复现 main.py 流程） 
    def run_single_question(self, item: Dict) -> Dict:
        """
        对单道 MCQ 题目运行完整的 MEDAR-QA 推理流程。
        流程与 main.py 的 main() 函数严格对齐。
        """
        question = format_mcq_question(item)
        print(f"问题：{question}")
        #  步骤 1：Agent A 问题分析与实体提取 
        agent_a_result     = self.agent_a.run(question)
        fod                = agent_a_result.get("frame_of_discernment", ["H", "¬H"])
        question_pico_data = agent_a_result.get("pico_elements", {})

        #  多轮检索闭环 
        all_evidence      : List[Dict] = []
        retrieval_history : List[Dict] = []
        current_round     : int        = 1

        # 初始化 AgentE（每题独立实例，防止推理历史污染）
        agent_e = AgentE()

        # 用于保存最后一轮的 AgentC/D 结果（循环外使用）
        agent_c_result : Dict = {}
        agent_d_result : Dict = {
            "final_decision": {
                "decision"  : "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "reason"    : "未获得决策结果"
            },
            "fusion_result"      : {},
            "belief_plausibility": {
                "hypothesis_positive": {"belief": 0, "plausibility": 1, "uncertainty_interval": 1},
                "hypothesis_negative": {"belief": 0, "plausibility": 1, "uncertainty_interval": 1}
            }
        }

        while current_round <= MAX_ROUNDS:

            #  步骤 2：知识检索 
            retrieval_result = retrieve_process(question, agent_a_result)
            all_evidence.extend(retrieval_result)
            retrieval_history.append({
                "round"          : current_round,
                "evidence_count" : len(retrieval_result),
                "total_evidence" : len(all_evidence)
            })

            #  步骤 3：Agent B PICO 提取与研究类型分类 
            agent_b_result = self.agent_b.run(question, all_evidence)

            #  步骤 4：Agent C 证据可靠性评估与 BPA 计算 
            contextual_question = f"原问题{question}\n当前识别框架为{fod}。"
            agent_c_result = self.agent_c.run(
                hypothesis           = contextual_question,
                agent_b_result       = agent_b_result,
                question_pico        = question_pico_data,
                frame_of_discernment = fod,          # 与 main.py 一致
                verbose              = False
            )

            #  步骤 5：Agent D 多证据融合与决策 
            bpa_list = agent_c_result.get("bpa_list", [])

            if not bpa_list:
                agent_d_result = {
                    "note"          : "No valid BPA for fusion",
                    "final_decision": {
                        "decision"  : "INSUFFICIENT_EVIDENCE",
                        "confidence": 0.0,
                        "reason"    : "没有足够的有效证据进行推理"
                    },
                    "fusion_result"      : {},
                    "belief_plausibility": {
                        "hypothesis_positive": {"belief": 0, "plausibility": 1, "uncertainty_interval": 1},
                        "hypothesis_negative": {"belief": 0, "plausibility": 1, "uncertainty_interval": 1}
                    }
                }
            else:
                agent_d_result = self.agent_d.run(question, fod, bpa_list)

            #  记录本轮推理到 Agent E（与 main.py 一致） 
            bpa_summary = {
                "average_support"    : sum(b.get("support_hypothesis", 0) for b in bpa_list) / len(bpa_list) if bpa_list else 0,
                "average_against"    : sum(b.get("against_hypothesis",  0) for b in bpa_list) / len(bpa_list) if bpa_list else 0,
                "average_uncertainty": sum(b.get("uncertainty", 0)         for b in bpa_list) / len(bpa_list) if bpa_list else 0,
                "bpa_count"          : len(bpa_list)
            }
            agent_e.add_reasoning_round(
                round_num      = current_round,
                evidence_count = len(retrieval_result),
                bpa_summary    = bpa_summary,
                note           = f"第{current_round}轮推理完成"
            )

            #  完备性分析（与 main.py 一致） 
            fused_bpa = agent_d_result.get("fusion_result", {}).get("fused_bpa", {
                "support_hypothesis": 0, "against_hypothesis": 0, "uncertainty": 1.0
            })
            belief_pl = agent_d_result.get("belief_plausibility", {
                "hypothesis_positive": {"belief": 0, "plausibility": 1, "uncertainty_interval": 1},
                "hypothesis_negative": {"belief": 0, "plausibility": 1, "uncertainty_interval": 1}
            })
            conflict_coef = agent_d_result.get("fusion_result", {}).get("conflict_coefficient", 0)

            completeness_result = self.controller.analyze_completeness(
                fused_bpa, belief_pl, conflict_coef
            )

            if not completeness_result["should_continue"] or current_round >= MAX_ROUNDS:
                break

            current_round += 1

        #  步骤 6：Agent E 最终报告生成（与 main.py 完全对齐） 

        # 注入 Agent D 逻辑追踪（与 main.py 一致）
        agent_d_logic_note = (
            f"[Agent D Logic Trace]:\n"
            f"- System Decision: {agent_d_result.get('final_decision', {}).get('decision')}\n"
            f"- Evidence Mapping: Agent D identified specific evidence IDs as supporting the decision.\n"
            f"  (Agent E MUST verify if this mapping aligns with clinical symptoms)."
        )
        agent_e.add_reasoning_round(99, 0, {}, note=agent_d_logic_note)

        # 构建增强型证据列表（Enhanced Evidence List）
        enhanced_evidence_input: List[Dict] = []
        c_evaluations = agent_c_result.get("evaluations", [])

        if c_evaluations:
            for ev in c_evaluations:
                ev_data = ev.get("evaluation", {})
                rich_content = ev_data.get("content_for_generator")
                if not rich_content:
                    rich_content = f"[Raw Snippet]: {ev_data.get('processed_input_snippet', 'N/A')}"
                enhanced_evidence_input.append({
                    "source"  : ev.get("source_type", "Unknown"),
                    "content" : rich_content,
                    "score"   : ev_data.get("bpa_components", {}).get("support_hypothesis", 0),
                    "type"    : "analyzed_report",
                    "metadata": ev.get("metadata", {})
                })
        else:
            # 回退到原始证据
            enhanced_evidence_input = all_evidence

        final_decision = agent_d_result.get("final_decision", {
            "decision"  : "UNCERTAIN",
            "confidence": 0.0,
            "reason"    : "未获得决策结果"
        })
        fusion_result = agent_d_result.get("fusion_result", {})

        agent_e_result = agent_e.run(
            question       = question,
            final_decision = final_decision,
            fusion_result  = fusion_result,
            evidence_list  = enhanced_evidence_input
        )
        print(f"Agent E 最终报告：{agent_e_result}\n")
        return {
            "question"            : question,
            "agent_a_analysis"    : agent_a_result,
            "agent_d_fusion"      : agent_d_result,
            "final_report"        : agent_e_result,
            "total_rounds"        : current_round,
            "retrieval_history"   : retrieval_history,
            "total_evidence_count": len(enhanced_evidence_input)
        }

    #  批量评估主流程 
    def run(self):
        data      = self.load_data()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        category_stats: Dict[str, Dict] = {}

        print("\n" + "=" * 70)
        print("MEDAR-QA  MedQA 数据集批量评估")
        print("=" * 70)
        print(f"数据路径   : {DATA_PATH}")
        print(f"测试题数   : {len(data)}")
        print(f"最大轮次   : {MAX_ROUNDS}")
        print(f"输出目录   : {OUTPUT_DIR}")
        print("=" * 70 + "\n")

        for i, item in enumerate(tqdm(data, desc="评估进度")):
            real_idx = item.get("realidx", i)
            try:
                print(f"\n[{i+1}/{len(data)}] 题目 #{real_idx}")

                response     = self.run_single_question(item)
                final_report = response.get("final_report", {})
                print(f"最终报告: {final_report}")
                predicted    = extract_predicted_answer(final_report)
                ground_truth = item["answer_idx"]
                is_correct   = (predicted.upper() == ground_truth.upper()) if predicted else False

                if is_correct:
                    self.correct_count += 1
                self.total_count += 1

                # 分类统计（step1 / step2&3 等）
                meta = item.get("meta_info", "unknown")
                if meta not in category_stats:
                    category_stats[meta] = {"correct": 0, "total": 0}
                category_stats[meta]["total"] += 1
                if is_correct:
                    category_stats[meta]["correct"] += 1

                # 记录结果
                self.results.append({
                    "realidx"             : real_idx,
                    "question"            : item["question"],
                    "options"             : item["options"],
                    "ground_truth"        : ground_truth,
                    "predicted"           : predicted,
                    "is_correct"          : is_correct,
                    "meta_info"           : meta,
                    "reasoning"           : (
                        final_report.get("reasoning", "")
                        or final_report.get("direct_answer", "")
                    )[:300],
                    "total_rounds"        : response.get("total_rounds", 1),
                    "total_evidence_count": response.get("total_evidence_count", 0)
                })

                acc  = self.correct_count / self.total_count * 100
                flag = "✓" if is_correct else "✗"
                print(f"  正确答案: {ground_truth}  |  预测答案: {predicted}  |  {flag}")
                print(f"  累计准确率: {self.correct_count}/{self.total_count} = {acc:.1f}%")

                # 定期保存检查点
                if (i + 1) % SAVE_INTERVAL == 0:
                    self._save_checkpoint(timestamp)

            except Exception as e:
                import traceback
                print(f"  [错误] 题目 #{real_idx}: {e}")
                traceback.print_exc()
                self.results.append({
                    "realidx"   : real_idx,
                    "error"     : str(e),
                    "is_correct": False,
                    "meta_info" : item.get("meta_info", "unknown")
                })
                self.total_count += 1

        # 保存最终结果
        self._save_final(timestamp, category_stats)

    #  保存工具 
    def _save_checkpoint(self, timestamp: str):
        acc    = self.correct_count / self.total_count if self.total_count > 0 else 0
        output = os.path.join(OUTPUT_DIR, f"medqa_{timestamp}_checkpoint.json")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(
                {"accuracy": f"{acc*100:.2f}%", "results": self.results},
                f, ensure_ascii=False, indent=2
            )
        print(f"  [检查点已保存]  {output}")

    def _save_final(self, timestamp: str, category_stats: Dict):
        accuracy = self.correct_count / self.total_count if self.total_count > 0 else 0

        summary = {
            "timestamp"     : timestamp,
            "config"        : {
                "data_path" : DATA_PATH,
                "max_rounds": MAX_ROUNDS,
                "test_limit": TEST_LIMIT
            },
            "total"         : self.total_count,
            "correct"       : self.correct_count,
            "accuracy"      : f"{accuracy*100:.2f}%",
            "category_stats": {
                k: {
                    **v,
                    "accuracy": f"{v['correct']/v['total']*100:.1f}%" if v["total"] > 0 else "N/A"
                }
                for k, v in category_stats.items()
            },
            "results"       : self.results
        }

        output = os.path.join(OUTPUT_DIR, f"medqa_{timestamp}.json")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print("评 估 完 成")
        print("=" * 70)
        print(f"总题数  : {self.total_count}")
        print(f"正确数  : {self.correct_count}")
        print(f"准确率  : {accuracy*100:.2f}%")
        print("\n分类准确率:")
        for cat, stats in category_stats.items():
            cat_acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {cat:20s}: {stats['correct']:3d}/{stats['total']:3d} = {cat_acc:.1f}%")
        print(f"\n结果文件: {output}")
        print("=" * 70)


# ------------------------------------------------------------------
if __name__ == "__main__":
    evaluator = MedQAEvaluator()
    evaluator.run()