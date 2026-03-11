"""
MEDAR-QA PubMedQA 批量测试脚本（完整流程版）
完全复刻 main.py 的推理流程，批量处理 PubMedQA 数据集。

与 main.py 的唯一区别：
  - 从 JSON 数据文件加载问题 + CONTEXTS（替代手动输入）
  - 在启动时对 LLM 客户端和 AgentC 打补丁，解决 Qwen3 think 模式下
    max_tokens 不足导致输出为空的问题，无需修改任何核心代码。

数据格式：pubmedqa_hard.json / pubmedqa_sample.json
"""

# ============================================================
# ★ 运行时补丁（必须在 import agents / llm_client 之前声明，
#   实际在下方 _apply_patches() 中被调用）
# ============================================================
import json
import os
import re
import time
import traceback
from datetime import datetime
from typing import Dict, List

# ==================== 用户配置 ====================
DATA_PATH     = "data/pubmedqa_sample.json"
OUTPUT_DIR    = "TEST_RESULTS/pubmedqa_batch"
TEST_LIMIT    = 500          # None = 全部；整数 = 只跑前 N 条
MAX_ROUNDS    = 1             # 与 main.py 保持一致
SAVE_INTERVAL = 100            # 每隔多少条保存一次中间结果

# Qwen3 等思考模型专用：调高 max_tokens 防止 think 块耗尽 token 限额
AGENT_C_MAX_TOKENS = 12000    # think 块 ~5000 + JSON 输出 ~1000，留足余量
AGENT_E_MAX_TOKENS = 3000     # 与 agents.py 原始值保持一致
# =================================================


# ----------------------------------------------------------
# 补丁函数定义（在 import agents 之后调用 _apply_patches()）
# ----------------------------------------------------------
def _patched_remove_think_tags(self, text: str) -> str:
    """
    增强版 _remove_think_tags:
    当 </think> 后无内容（max_tokens 被 think 块耗尽）时,
    回退到 think 块内部用状态机提取最大合法 JSON 对象。
    """
    if not text:
        return text

    def _scan_largest_json(s: str) -> str:
        """状态机扫描：找出字符串中最大的完整 JSON 对象"""
        best, n, i = "", len(s), 0
        while i < n:
            if s[i] != '{':
                i += 1
                continue
            depth, in_str, esc, j = 0, False, False, i
            while j < n:
                ch = s[j]
                if esc:              esc = False
                elif ch == '\\' and in_str: esc = True
                elif ch == '"':      in_str = not in_str
                elif not in_str:
                    if ch == '{':    depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = s[i:j + 1]
                            if len(candidate) > len(best):
                                best = candidate
                            break
                j += 1
            i += 1
        return best

    # 情况1: </think> 正常闭合
    if '</think>' in text:
        after = text.split('</think>')[-1].strip()
        if after:
            return after
        # </think> 后为空 → think 块耗尽了 max_tokens → 从 think 内提取 JSON
        print("⚠️  think 块耗尽 max_tokens，尝试从 think 内容中提取 JSON...")
        think_body = re.sub(r'</?think>', '', text).strip()
        found = _scan_largest_json(think_body)
        if found:
            return found
        print("⚠️  think 内部也未找到合法 JSON。建议增大 max_tokens 或设置 DISABLE_THINKING=True")
        return ''

    # 情况2: 只有 <think> 被截断（没有 </think>）
    if '<think>' in text:
        before = text.split('<think>')[0].strip()
        if before:
            return before
        think_body = text.split('<think>', 1)[1]
        found = _scan_largest_json(think_body)
        return found if found else ''

    # 情况3: 无 think 标签
    return text.strip()


def _patched_agent_c_call_llm(self, prompt: str, temperature: float = 0.1,
                               max_retries: int = 3, retry_delay: float = 5.0) -> str:
    """
    增强版 AgentC._call_llm_api:
    - max_tokens 提升到 AGENT_C_MAX_TOKENS，防止 think 块截断
    - 加入指数退避重试，应对短暂网络抖动
    """
    from llm_client import call_llm
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = call_llm(prompt, temperature=temperature,
                              max_tokens=AGENT_C_MAX_TOKENS)
            if result:
                return result
            raise ValueError("LLM returned empty response")
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = retry_delay * (2 ** (attempt - 1))
                print(f"[Agent C] 第{attempt}次调用失败，{wait:.0f}s 后重试... ({e})")
                time.sleep(wait)
    print(f"[Agent C] 连续 {max_retries} 次失败，跳过本条评估: {last_error}")
    return ""


def _apply_patches():
    """在核心模块导入后打上运行时补丁（不修改任何源文件）"""
    import llm_client
    import agents

    # 补丁1: 修复 _remove_think_tags（修复 think 块耗尽 max_tokens 时输出为空的问题）
    llm_client.OpenAICompatibleClient._remove_think_tags = _patched_remove_think_tags
    print("✓ [Patch] llm_client._remove_think_tags 已增强")

    # 补丁2: 提升 AgentC max_tokens + 重试（已有重试逻辑则仅替换 max_tokens）
    agents.AgentC._call_llm_api = _patched_agent_c_call_llm
    print(f"✓ [Patch] AgentC._call_llm_api → max_tokens={AGENT_C_MAX_TOKENS}，带重试")


# ----------------------------------------------------------
# 应用补丁后再导入依赖 agent 的模块
# ----------------------------------------------------------
from tqdm import tqdm
_apply_patches()

from agents import AgentA, AgentB, AgentC, AgentD, AgentE, CompletenessController, extract_json_from_response
from retriever import retrieve_process


# ----------------------------------------------------------
# 工具函数
# ----------------------------------------------------------
def normalize_answer(text: str) -> str:
    """标准化为 yes / no / maybe"""
    if not text:
        return ""
    t = text.lower().strip()
    if t in ("yes", "positive", "support"):    return "yes"
    if t in ("no", "negative", "against"):     return "no"
    if t in ("maybe", "uncertain", "insufficient_evidence"): return "maybe"
    return t


def extract_answer(response: Dict) -> str:
    """从 AgentE 响应中提取最终答案"""
    ans = normalize_answer(response.get("answer", ""))
    if ans in ("yes", "no", "maybe"):
        return ans
    # 降级：从 reasoning 文本中匹配
    reasoning = response.get("reasoning", "").lower()
    for kw in ("yes", "no", "maybe"):
        if kw in reasoning:
            return kw
    return ""


# ----------------------------------------------------------
# 核心流程（完全复刻 main.py）
# ----------------------------------------------------------
def run_pipeline_single(question: str, context: str,
                        agent_a: AgentA, agent_b: AgentB,
                        agent_c: AgentC, agent_d: AgentD,
                        agent_e: AgentE,
                        controller: CompletenessController) -> Dict:
    """
    完整复刻 main.py 的单条推理流程，逐步骤执行 A→B→C→D→E。
    """
    # ── Step 1: Agent A ──────────────────────────────────────────────────────
    agent_a_result = agent_a.run(question)
    fod = agent_a_result.get("frame_of_discernment", ["H", "¬H"])

    # 多轮检索初始化
    current_round = 1
    all_evidence  = []

    # 用户上下文（CONTEXTS）作为初始证据片段，与 main.py 保持一致
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

    # 用来接收各轮结果的变量（若 max_rounds=1 则只跑一轮）
    agent_c_result = {}
    agent_d_result = {
        "final_decision": {"decision": "INSUFFICIENT_EVIDENCE", "confidence": 0.0,
                           "reason": "没有足够的有效证据"},
        "fusion_result": {}, "belief_plausibility": {}
    }

    while current_round <= MAX_ROUNDS:

        # ── Step 2: 知识检索 ─────────────────────────────────────────────────
        retrieval_result = retrieve_process(question, agent_a_result)
        all_evidence.extend(retrieval_result)

        # ── Step 3: Agent B ──────────────────────────────────────────────────
        agent_b_result = agent_b.run(question, all_evidence)

        # ── Step 4: Agent C ──────────────────────────────────────────────────
        contextual_question = f"原问题{question}\n当前识别框架为{fod}。"
        question_pico       = agent_a_result.get("pico_elements", {})
        agent_c_result = agent_c.run(
            hypothesis=contextual_question,
            agent_b_result=agent_b_result,
            question_pico=question_pico,
            frame_of_discernment=fod,
            verbose=False
        )

        # ── Step 5: Agent D ──────────────────────────────────────────────────
        bpa_list = agent_c_result.get("bpa_list", [])
        if not bpa_list:
            agent_d_result = {
                "final_decision": {"decision": "INSUFFICIENT_EVIDENCE", "confidence": 0.0,
                                   "reason": "没有足够BPA"},
                "fusion_result": {}, "belief_plausibility": {}
            }
        else:
            agent_d_result = agent_d.run(question, fod, bpa_list)

        # 记录本轮到 Agent E 的推理历史
        bpa_summary = {
            "average_support":   sum(b.get("support_hypothesis", 0) for b in bpa_list) / len(bpa_list) if bpa_list else 0,
            "average_against":   sum(b.get("against_hypothesis", 0) for b in bpa_list) / len(bpa_list) if bpa_list else 0,
            "average_uncertainty": sum(b.get("uncertainty", 0) for b in bpa_list) / len(bpa_list) if bpa_list else 0,
            "bpa_count": len(bpa_list)
        }
        agent_e.add_reasoning_round(
            round_num=current_round,
            evidence_count=len(retrieval_result),
            bpa_summary=bpa_summary,
            note=f"第{current_round}轮推理完成"
        )

        # 完备性分析（与 main.py 完全一致）
        fused_bpa    = agent_d_result.get("fusion_result", {}).get("fused_bpa", {
            "support_hypothesis": 0, "against_hypothesis": 0, "uncertainty": 1.0})
        belief_pl    = agent_d_result.get("belief_plausibility", {
            "hypothesis_positive": {"belief": 0, "plausibility": 1, "uncertainty_interval": 1},
            "hypothesis_negative": {"belief": 0, "plausibility": 1, "uncertainty_interval": 1}})
        conflict_coef = agent_d_result.get("fusion_result", {}).get("conflict_coefficient", 0)
        completeness  = controller.analyze_completeness(fused_bpa, belief_pl, conflict_coef)

        if not completeness["should_continue"] or current_round >= MAX_ROUNDS:
            break
        current_round += 1

    # ── Step 6: Agent E ──────────────────────────────────────────────────────
    agent_d_logic_note = (
        f"[Agent D Logic Trace]: System Decision: "
        f"{agent_d_result.get('final_decision', {}).get('decision')}; "
        f"Agent E MUST verify alignment with clinical symptoms."
    )
    agent_e.add_reasoning_round(99, 0, {}, note=agent_d_logic_note)

    # 构建增强型证据列表（与 main.py 完全一致）
    c_evaluations = agent_c_result.get("evaluations", [])
    if c_evaluations:
        enhanced = []
        for ev in c_evaluations:
            ev_data     = ev.get("evaluation", {})
            rich_content = ev_data.get("content_for_generator") or \
                           f"[Raw Snippet]: {ev_data.get('processed_input_snippet', 'N/A')}"
            enhanced.append({
                "source":   ev.get("source_type", "Unknown"),
                "content":  rich_content,
                "score":    ev_data.get("bpa_components", {}).get("support_hypothesis", 0),
                "type":     "analyzed_report",
                "metadata": ev.get("metadata", {})
            })
    else:
        enhanced = all_evidence

    final_decision = agent_d_result.get("final_decision", {
        "decision": "UNCERTAIN", "confidence": 0.0, "reason": "未获得决策结果"})
    fusion_result  = agent_d_result.get("fusion_result", {})

    agent_e_result = agent_e.run(
        question=question,
        final_decision=final_decision,
        fusion_result=fusion_result,
        evidence_list=enhanced
    )
    return agent_e_result


# ----------------------------------------------------------
# 批量评估器
# ----------------------------------------------------------
class PubMedQAEvaluator:

    def __init__(self):
        self.results       = []
        self.correct_count = 0
        self.total_count   = 0

        # A/B/C/D 实例复用（无跨条目状态），AgentE 每条重建
        self.agent_a    = AgentA()
        self.agent_b    = AgentB()
        self.agent_c    = AgentC()
        self.agent_d    = AgentD()
        self.controller = CompletenessController()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 数据加载 ─────────────────────────────────────────────────────────────
    def load_data(self) -> List[Dict]:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data = []
        for pmid, item in raw.items():
            item["pmid"] = pmid
            data.append(item)
            if TEST_LIMIT and len(data) >= TEST_LIMIT:
                break
        print(f"共加载 {len(data)} 条测试数据")
        return data

    # ── 单条测试 ─────────────────────────────────────────────────────────────
    def run_single(self, item: Dict) -> Dict:
        pmid         = item.get("pmid", "unknown")
        question     = item.get("QUESTION", "")
        contexts     = item.get("CONTEXTS", [])
        ground_truth = item.get("final_decision", "")

        # CONTEXTS 拼接为文本（与 main.py 示例 context 格式一致）
        context = "\n".join(contexts)

        try:
            # AgentE 每条重建，隔离推理历史
            agent_e = AgentE()

            result    = run_pipeline_single(
                question, context,
                self.agent_a, self.agent_b,
                self.agent_c, self.agent_d,
                agent_e, self.controller
            )
            predicted  = extract_answer(result)
            is_correct = (predicted == normalize_answer(ground_truth))

            return {
                "pmid":         pmid,
                "question":     question,
                "ground_truth": ground_truth,
                "predicted":    predicted,
                "is_correct":   is_correct,
                "confidence":   result.get("confidence_score"),
                "reasoning":    result.get("reasoning", ""),
                "error":        None
            }

        except Exception:
            return {
                "pmid":         pmid,
                "question":     question,
                "ground_truth": ground_truth,
                "predicted":    None,
                "is_correct":   False,
                "confidence":   None,
                "reasoning":    "",
                "error":        traceback.format_exc()
            }

    # ── 批量评估 ─────────────────────────────────────────────────────────────
    def run(self):
        print("=" * 80)
        print("PubMedQA 批量评估  —  完整 MEDAR-QA 流程")
        print("=" * 80)

        data = self.load_data()

        for i, item in enumerate(tqdm(data, desc="评估进度")):
            rec = self.run_single(item)
            self.results.append(rec)
            self.total_count += 1
            if rec["is_correct"]:
                self.correct_count += 1

            acc     = self.correct_count / self.total_count * 100
            flag    = "✓" if rec["is_correct"] else "✗"
            gt_s    = str(rec["ground_truth"] or "")
            pred_s  = str(rec["predicted"]    or "N/A")
            err_s   = (" ERR: " + rec["error"].splitlines()[0][:80]) if rec["error"] else ""
            tqdm.write(
                f"[{i+1:>5}/{len(data)}] {flag}  PMID={rec['pmid']:<10}"
                f"  GT={gt_s:<5}  Pred={pred_s:<5}  Acc={acc:.1f}%{err_s}"
            )

            if (i + 1) % SAVE_INTERVAL == 0:
                self._save(interim=True)

        self._save(interim=False)
        self._print_summary()

    # ── 保存结果 ─────────────────────────────────────────────────────────────
    def _save(self, interim: bool):
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix  = "interim_" if interim else "final_"
        fpath   = f"{OUTPUT_DIR}/pubmedqa_{prefix}{ts}.json"
        acc_val = round(self.correct_count / self.total_count * 100, 4) if self.total_count else 0

        out = {
            "meta": {
                "timestamp":     ts,
                "data_path":     DATA_PATH,
                "test_limit":    TEST_LIMIT,
                "max_rounds":    MAX_ROUNDS,
                "agent_c_max_tokens": AGENT_C_MAX_TOKENS,
                "total_count":   self.total_count,
                "correct_count": self.correct_count,
                "accuracy":      acc_val
            },
            "results": [
                {k: v for k, v in r.items() if k != "full_result"}
                for r in self.results
            ]
        }
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        tqdm.write(f"\n  → 已保存：{fpath}\n")

    # ── 摘要统计 ─────────────────────────────────────────────────────────────
    def _print_summary(self):
        print("\n" + "=" * 80)
        print("评估摘要")
        print("=" * 80)
        total   = self.total_count
        correct = self.correct_count
        print(f"总数量   : {total}")
        print(f"正确数量 : {correct}")
        print(f"准确率   : {correct/total*100:.2f}%" if total else "准确率   : N/A")

        print("\n按标准答案分类：")
        for label in ("yes", "no", "maybe"):
            t = sum(1 for r in self.results if normalize_answer(r["ground_truth"]) == label)
            c = sum(1 for r in self.results if normalize_answer(r["ground_truth"]) == label and r["is_correct"])
            if t:
                print(f"  {label.capitalize():<6}: {c}/{t}  ({c/t*100:.1f}%)")

        print("\n预测分布：")
        from collections import Counter
        dist = Counter(r.get("predicted") or "(空)" for r in self.results)
        for k, v in dist.most_common():
            print(f"  {str(k):<12}: {v}")

        err_n = sum(1 for r in self.results if r["error"])
        if err_n:
            print(f"\n运行时异常数量 : {err_n}")
        print("=" * 80)


if __name__ == "__main__":
    evaluator = PubMedQAEvaluator()
    evaluator.run()
