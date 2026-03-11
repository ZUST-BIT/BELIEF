"""
MEDAR-QA PubMedQA 批量测试脚本
完全复刻 main.py 的推理流程，批量处理 PubMedQA 数据集并评估准确率。
数据格式：pubmedqa_hard.json
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

from agents import AgentA, AgentB, AgentC, AgentD, AgentE, CompletenessController, extract_json_from_response
from retriever import retrieve_process


# ==================== 配置参数 ====================
DATA_PATH   = "data/pubmedqa_sample.json"       # PubMedQA 数据集路径
OUTPUT_DIR  = "TEST_RESULTS/pubmedqa_batch"   # 结果输出目录
TEST_LIMIT  = None                             # 测试数量，None 表示全部
MAX_ROUNDS  = 1                                # 最大检索轮次（与 main.py 保持一致）
SAVE_INTERVAL = 500                             # 每隔多少条自动保存一次中间结果
# ================================================


def normalize_decision(decision: str) -> str:
    """标准化 decision 为 yes / no / maybe"""
    if not decision:
        return ""
    d = decision.lower().strip()
    if d in ("yes", "positive", "support", "accept"):
        return "yes"
    if d in ("no", "negative", "against", "reject"):
        return "no"
    if d in ("maybe", "uncertain", "insufficient_evidence", "inconclusive"):
        return "maybe"
    return d


def extract_answer(response: Dict) -> str:
    """从 AgentE 响应中提取 yes/no/maybe 答案"""
    # 优先从 answer 字段获取
    ans = response.get("answer", "")
    if ans and ans.lower() in ("yes", "no", "maybe"):
        return ans.lower()

    # 从 reasoning 中尝试提取（降级处理）
    reasoning = response.get("reasoning", "").lower()
    for kw in ("yes", "no", "maybe"):
        if kw in reasoning:
            return kw
    return ""


def run_pipeline(question: str, context: str,
                 agent_a: AgentA, agent_b: AgentB,
                 agent_c: AgentC, agent_d: AgentD,
                 agent_e: AgentE,
                 controller: CompletenessController) -> Dict:
    """
    完整复刻 main.py 的单条推理流程。
    返回 AgentE 的输出结果字典。
    """

    # ── Step 1: Agent A ──────────────────────────────
    agent_a_result = agent_a.run(question)
    fod = agent_a_result.get("frame_of_discernment", ["H", "¬H"])

    # 多轮检索闭环初始化
    max_rounds    = MAX_ROUNDS
    current_round = 1
    all_evidence  = []

    # 用户提供的上下文作为初始证据片段
    if context and context.strip():
        all_evidence.append({
            "source": "user_provided_context",
            "content": context.strip(),
            "score": 1.0,
            "type": "user_context",
            "metadata": {
                "is_user_provided": True,
                "description": "用户提供的背景上下文信息"
            }
        })

    # 保存每轮的 agent_c / agent_d 结果，供 Step 6 使用
    agent_c_result = {}
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

    while current_round <= max_rounds:

        # ── Step 2: 知识检索 ─────────────────────────
        retrieval_result = retrieve_process(question, agent_a_result)
        all_evidence.extend(retrieval_result)

        # ── Step 3: Agent B ──────────────────────────
        agent_b_result = agent_b.run(question, all_evidence)

        # ── Step 4: Agent C ──────────────────────────
        contextual_question = f"原问题{question}\n当前识别框架为{fod}。"
        question_pico_data  = agent_a_result.get("pico_elements", {})
        agent_c_result = agent_c.run(
            hypothesis=contextual_question,
            agent_b_result=agent_b_result,
            question_pico=question_pico_data,
            frame_of_discernment=fod,   # 与 main.py 完全一致
            verbose=False
        )

        # ── Step 5: Agent D ──────────────────────────
        bpa_list = agent_c_result.get("bpa_list", [])

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
            agent_d_result = agent_d.run(question, fod, bpa_list)

        # 记录本轮推理到 Agent E
        bpa_summary = {
            "average_support":     sum(b.get("support_hypothesis", 0)  for b in bpa_list) / len(bpa_list) if bpa_list else 0,
            "average_against":     sum(b.get("against_hypothesis", 0)  for b in bpa_list) / len(bpa_list) if bpa_list else 0,
            "average_uncertainty": sum(b.get("uncertainty", 0)         for b in bpa_list) / len(bpa_list) if bpa_list else 0,
            "bpa_count": len(bpa_list)
        }
        agent_e.add_reasoning_round(
            round_num=current_round,
            evidence_count=len(retrieval_result),
            bpa_summary=bpa_summary,
            note=f"第{current_round}轮推理完成"
        )

        # ── 完备性分析 ────────────────────────────────
        fused_bpa = agent_d_result.get("fusion_result", {}).get("fused_bpa", {
            "support_hypothesis": 0,
            "against_hypothesis": 0,
            "uncertainty": 1.0
        })
        belief_pl = agent_d_result.get("belief_plausibility", {
            "hypothesis_positive": {"belief": 0, "plausibility": 1, "uncertainty_interval": 1},
            "hypothesis_negative": {"belief": 0, "plausibility": 1, "uncertainty_interval": 1}
        })
        conflict_coef = agent_d_result.get("fusion_result", {}).get("conflict_coefficient", 0)
        completeness_result = controller.analyze_completeness(fused_bpa, belief_pl, conflict_coef)

        if not completeness_result["should_continue"]:
            break
        if current_round >= max_rounds:
            break
        current_round += 1

    # ── Step 6: Agent E ──────────────────────────────
    # 与 main.py 完全一致的 agent_d_logic_note
    agent_d_logic_note = (
        f"[Agent D Logic Trace]: "
        f"System Decision: {agent_d_result.get('final_decision', {}).get('decision')}; "
        f"Evidence Mapping: Agent D identified specific evidence IDs as supporting the decision. "
        f"(Agent E MUST verify if this mapping aligns with clinical symptoms)."
    )
    agent_e.add_reasoning_round(99, 0, {}, note=agent_d_logic_note)

    # 构建增强型证据列表（与 main.py 完全一致）
    enhanced_evidence_input = []
    c_evaluations = agent_c_result.get("evaluations", [])
    if c_evaluations:
        for ev in c_evaluations:
            ev_data     = ev.get("evaluation", {})
            rich_content = ev_data.get("content_for_generator")
            if not rich_content:
                rich_content = f"[Raw Snippet]: {ev_data.get('processed_input_snippet', 'N/A')}"
            enhanced_evidence_input.append({
                "source":   ev.get("source_type", "Unknown"),
                "content":  rich_content,
                "score":    ev_data.get("bpa_components", {}).get("support_hypothesis", 0),
                "type":     "analyzed_report",
                "metadata": ev.get("metadata", {})
            })
    else:
        enhanced_evidence_input = all_evidence

    final_decision = agent_d_result.get("final_decision", {
        "decision": "UNCERTAIN",
        "confidence": 0.0,
        "reason": "未获得决策结果"
    })
    fusion_result  = agent_d_result.get("fusion_result", {})

    agent_e_result = agent_e.run(
        question=question,
        final_decision=final_decision,
        fusion_result=fusion_result,
        evidence_list=enhanced_evidence_input
    )

    return agent_e_result


class PubMedQABatchEvaluator:
    """PubMedQA 批量评估器"""

    def __init__(self):
        self.results      = []
        self.correct_count = 0
        self.total_count   = 0

        # 与 main.py 一样，每条样本重用同一批智能体实例（仅 AgentE 按样本重建）
        self.agent_a    = AgentA()
        self.agent_b    = AgentB()
        self.agent_c    = AgentC()
        self.agent_d    = AgentD()
        self.controller = CompletenessController()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 数据加载 ───────────────────────────────────────────────────────────────
    def load_data(self) -> List[Dict]:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        data = []
        for pmid, item in raw_data.items():
            item["pmid"] = pmid
            data.append(item)
            if TEST_LIMIT and len(data) >= TEST_LIMIT:
                break
        return data

    # ── 单条测试 ───────────────────────────────────────────────────────────────
    def run_single(self, item: Dict) -> Dict:
        pmid         = item.get("pmid", "unknown")
        question     = item.get("QUESTION", "")
        contexts     = item.get("CONTEXTS", [])
        ground_truth = item.get("final_decision", "")      # 标准答案

        # 将 CONTEXTS 列表拼接为单一字符串（与 main.py 示例 context 格式一致）
        context = "\n".join(contexts)

        try:
            # 每条样本创建新的 AgentE 实例（保持推理历史隔离）
            agent_e = AgentE()

            result    = run_pipeline(
                question, context,
                self.agent_a, self.agent_b,
                self.agent_c, self.agent_d,
                agent_e, self.controller
            )
            predicted  = extract_answer(result)
            is_correct = (predicted == normalize_decision(ground_truth))

            return {
                "pmid":         pmid,
                "question":     question,
                "ground_truth": ground_truth,
                "predicted":    predicted,
                "reasoning":    result.get("reasoning", ""),
                "confidence":   result.get("confidence_score", None),
                "is_correct":   is_correct,
                "full_result":  result,
                "error":        None
            }

        except Exception as e:
            import traceback
            return {
                "pmid":         pmid,
                "question":     question,
                "ground_truth": ground_truth,
                "predicted":    None,
                "reasoning":    "",
                "confidence":   None,
                "is_correct":   False,
                "full_result":  None,
                "error":        traceback.format_exc()
            }

    # ── 批量评估 ───────────────────────────────────────────────────────────────
    def run_evaluation(self):
        print("=" * 80)
        print("PubMedQA 批量评估  —  完整 MEDAR-QA 流程")
        print("=" * 80)

        data = self.load_data()
        print(f"共加载 {len(data)} 条测试数据\n")

        for i, item in enumerate(tqdm(data, desc="测试进度")):
            record = self.run_single(item)
            self.results.append(record)
            self.total_count += 1
            if record["is_correct"]:
                self.correct_count += 1

            accuracy  = self.correct_count / self.total_count * 100
            status    = "✓" if record["is_correct"] else "✗"
            gt_str    = str(record["ground_truth"] or "")
            pred_str  = str(record["predicted"]    or "N/A")
            print(
                f"[{i+1:>4}/{len(data)}] {status}  PMID={record['pmid']}"
                f"  GT={gt_str:<5}  Pred={pred_str:<5}"
                f"  Acc={accuracy:.1f}%"
                + (f"  ERR: {str(record['error']).splitlines()[0][:80]}" if record["error"] else "")
            )

            # 定期保存中间结果
            if (i + 1) % SAVE_INTERVAL == 0:
                self.save_results(interim=True)

        self.save_results(interim=False)
        self.print_summary()

    # ── 结果保存 ───────────────────────────────────────────────────────────────
    def save_results(self, interim: bool = False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix    = "interim_" if interim else "final_"
        filename  = f"{OUTPUT_DIR}/pubmedqa_{prefix}{timestamp}.json"

        save_data = {
            "meta": {
                "timestamp":     timestamp,
                "data_path":     DATA_PATH,
                "test_limit":    TEST_LIMIT,
                "max_rounds":    MAX_ROUNDS,
                "total_count":   self.total_count,
                "correct_count": self.correct_count,
                "accuracy":      round(self.correct_count / self.total_count * 100, 4)
                                 if self.total_count > 0 else 0
            },
            "results": [
                {
                    "pmid":         r["pmid"],
                    "question":     r["question"],
                    "ground_truth": r["ground_truth"],
                    "predicted":    r["predicted"],
                    "is_correct":   r["is_correct"],
                    "confidence":   r["confidence"],
                    "reasoning":    r["reasoning"],
                    "error":        r["error"]
                }
                for r in self.results
            ]
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\n  → 结果已保存：{filename}")

    # ── 评估摘要 ───────────────────────────────────────────────────────────────
    def print_summary(self):
        print("\n" + "=" * 80)
        print("评估摘要")
        print("=" * 80)
        print(f"总数量   : {self.total_count}")
        print(f"正确数量 : {self.correct_count}")
        print(f"准确率   : {self.correct_count / self.total_count * 100:.2f}%"
              if self.total_count else "准确率   : N/A")

        # 分类统计
        print("\n按答案类型统计：")
        for label in ("yes", "no", "maybe"):
            total   = sum(1 for r in self.results if normalize_decision(r["ground_truth"]) == label)
            correct = sum(1 for r in self.results if normalize_decision(r["ground_truth"]) == label and r["is_correct"])
            if total > 0:
                print(f"  {label.capitalize():<6}: {correct}/{total}  ({correct/total*100:.1f}%)")

        # 预测分布
        print("\n预测答案分布：")
        for label in ("yes", "no", "maybe", ""):
            count = sum(1 for r in self.results if r.get("predicted") == label)
            if count:
                tag = label if label else "(空/未识别)"
                print(f"  {tag:<12}: {count}")

        # 错误数量
        error_count = sum(1 for r in self.results if r["error"] is not None)
        if error_count:
            print(f"\n运行时错误数量 : {error_count}")

        print("=" * 80)


def main():
    evaluator = PubMedQABatchEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
