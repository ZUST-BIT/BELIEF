"""
MEDAR-QA PubMedQA 批量测试脚本 v2（含直接LLM分支 + 最终聚合）
完全复刻 main.py 的最新推理流程，包括：
  - 直接LLM推理分支（ENABLE_DIRECT_LLM_BRANCH 开关控制）
  - 最终聚合智能体（当分支开启时启动，综合 DS 与 LLM 两条路径）
  - 三路答案同时评估（DS 分支 / 直接LLM 分支 / 最终聚合）

与 run_pubmedqa_full.py 的主要差异：
  - 引入 AgentDirectLLM、AgentFinalAggregator
  - run_pipeline_single 返回完整三路结果字典
  - PubMedQAEvaluator 同时跟踪三路准确率
  - 保存结果含 direct_llm_result / final_aggregated_result 字段

数据格式：pubmedqa_hard.json / pubmedqa_sample.json
"""

# ============================================================
# ★ 运行时补丁（必须在 import agents / llm_client 之前声明）
# ============================================================
import json
import os
import re
import time
import traceback
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

# ==================== 用户配置 ====================
DATA_PATH     = "data/pubmedqa_sample.json"
OUTPUT_DIR    = "TEST_RESULTS/pubmedqa_batch_v2"
TEST_LIMIT    = 100          # None = 全部；整数 = 只跑前 N 条
MAX_ROUNDS    = 1            # 与 main.py 保持一致
SAVE_INTERVAL = 10           # 每隔多少条保存一次中间结果

# 直接LLM分支开关
# True  → 在 DS 流程之外额外运行直接LLM推理，最后由聚合智能体综合两路结果
# False → 仅运行现有 DS 流程（与旧版 run_pubmedqa_full.py 等价）
ENABLE_DIRECT_LLM_BRANCH: bool = True

# Qwen3 等思考模型专用：调高 max_tokens 防止 think 块耗尽 token 限额
AGENT_C_MAX_TOKENS = 12000   # think 块 ~5000 + JSON 输出 ~1000，留足余量
AGENT_E_MAX_TOKENS = 3000    # 与 agents.py 原始值保持一致
# ==================================================


# ----------------------------------------------------------
# 补丁函数定义（在 import agents 之后调用 _apply_patches()）
# ----------------------------------------------------------
def _patched_remove_think_tags(self, text: str) -> str:
    """
    增强版 _remove_think_tags：
    当 </think> 后无内容（max_tokens 被 think 块耗尽）时，
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
                if esc:               esc = False
                elif ch == '\\' and in_str: esc = True
                elif ch == '"':       in_str = not in_str
                elif not in_str:
                    if ch == '{':     depth += 1
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
        # </think> 后为空 → 从 think 内提取 JSON
        print("⚠️  think 块耗尽 max_tokens，尝试从 think 内容中提取 JSON...")
        think_body = re.sub(r'</?think>', '', text).strip()
        found = _scan_largest_json(think_body)
        if found:
            return found
        print("⚠️  think 内部也未找到合法 JSON，建议增大 max_tokens 或关闭 thinking 模式")
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
    增强版 AgentC._call_llm_api：
    - max_tokens 提升到 AGENT_C_MAX_TOKENS，防止 think 块截断
    - 带指数退避重试，应对短暂网络抖动
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

    # 补丁2: 提升 AgentC max_tokens + 重试
    agents.AgentC._call_llm_api = _patched_agent_c_call_llm
    print(f"✓ [Patch] AgentC._call_llm_api → max_tokens={AGENT_C_MAX_TOKENS}，带重试")


# ----------------------------------------------------------
# 应用补丁后再导入依赖 agent 的模块
# ----------------------------------------------------------
from tqdm import tqdm
_apply_patches()

from agents import (AgentA, AgentB, AgentC, AgentD, AgentE,
                    CompletenessController, AgentDirectLLM, AgentFinalAggregator,
                    extract_json_from_response)
from retriever import retrieve_process


# ----------------------------------------------------------
# 工具函数
# ----------------------------------------------------------
def normalize_answer(text: str) -> str:
    """标准化为 yes / no / maybe"""
    if not text:
        return ""
    t = str(text).lower().strip()
    if t in ("yes", "positive", "support", "support_association"):
        return "yes"
    if t in ("no", "negative", "against", "refute_association"):
        return "no"
    if t in ("maybe", "uncertain", "insufficient_evidence", "inconclusive"):
        return "maybe"
    return t


def extract_ds_answer(agent_e_result: Dict) -> str:
    """从 Agent E（DS 分支）响应中提取答案 yes/no/maybe"""
    ans = normalize_answer(agent_e_result.get("answer", ""))
    if ans in ("yes", "no", "maybe"):
        return ans
    # 降级：从 reasoning 文本中匹配
    reasoning = str(agent_e_result.get("reasoning", "")).lower()
    for kw in ("yes", "no", "maybe"):
        if kw in reasoning:
            return kw
    return ""


def extract_direct_llm_answer(direct_llm_result: Optional[Dict]) -> str:
    """从直接LLM分支结果中提取答案（Yes/No 题：answer 字段）"""
    if not direct_llm_result:
        return ""
    ans = normalize_answer(direct_llm_result.get("answer", ""))
    if ans in ("yes", "no", "maybe"):
        return ans
    # 降级：从 reasoning 中匹配
    reasoning = str(direct_llm_result.get("reasoning", "")).lower()
    for kw in ("yes", "no", "maybe"):
        if kw in reasoning:
            return kw
    return ""


def extract_aggregated_answer(final_aggregated_result: Optional[Dict]) -> str:
    """从最终聚合结果中提取答案（final_answer 字段）"""
    if not final_aggregated_result:
        return ""
    ans = normalize_answer(final_aggregated_result.get("final_answer", ""))
    if ans in ("yes", "no", "maybe"):
        return ans
    # 降级：从 reasoning 中匹配
    reasoning = str(final_aggregated_result.get("reasoning", "")).lower()
    for kw in ("yes", "no", "maybe"):
        if kw in reasoning:
            return kw
    return ""


def _dedup_against_user_context(papers: list, user_context_text: str,
                                fingerprint_len: int = 60) -> list:
    """过滤与用户上下文高度重叠的检索论文，防止双重计数（与 main.py 保持一致）"""
    if not user_context_text:
        return papers
    ctx_lower = user_context_text.lower()
    unique = []
    for paper in papers:
        content = paper.get('content', '')
        is_dup = False
        for line in content.split('\n'):
            line_stripped = line.strip()
            if line_stripped.lower().startswith('summary:'):
                body = line_stripped[len('summary:'):].strip()
                if ':' in body[:20]:
                    body = body[body.index(':') + 1:].strip()
                fingerprint = body[:fingerprint_len].lower()
                if fingerprint and fingerprint in ctx_lower:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(paper)
    return unique


# ----------------------------------------------------------
# 核心流程（完全复刻 main.py，含新增两条分支）
# ----------------------------------------------------------
def run_pipeline_single(
        question: str,
        context: str,
        agent_a: AgentA,
        agent_b: AgentB,
        agent_c: AgentC,
        agent_d: AgentD,
        agent_e: AgentE,
        controller: CompletenessController,
        agent_direct_llm: Optional[AgentDirectLLM] = None,
        agent_final_agg: Optional[AgentFinalAggregator] = None,
        enable_direct_branch: bool = True,
) -> Dict:
    """
    完整复刻 main.py（含直接LLM分支与最终聚合）的单条推理流程。

    Returns:
        {
            "agent_e_result":           ...,   # DS 分支输出（Agent E 报告）
            "direct_llm_result":        ...,   # 直接LLM分支输出（None 若关闭）
            "final_aggregated_result":  ...,   # 最终聚合输出（None 若关闭）
            "task_mode":                ...,   # SELECTION / OPEN_REASONING
        }
    """
    # ── Step 1: Agent A ──────────────────────────────────────────────────────
    agent_a_result = agent_a.run(question)
    fod       = agent_a_result.get("frame_of_discernment", ["H", "¬H"])
    task_mode = agent_a_result.get("task_mode", "OPEN_REASONING")

    # 多轮检索初始化
    current_round    = 1
    all_evidence     = []
    retrieval_history = []

    # 用户上下文（CONTEXTS）作为初始证据片段
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

    # 各轮结果占位
    agent_b_result = {}
    agent_c_result = {}
    agent_d_result = {
        "final_decision": {"decision": "INSUFFICIENT_EVIDENCE",
                           "confidence": 0.0, "reason": "没有足够的有效证据"},
        "fusion_result": {}, "belief_plausibility": {}
    }
    direct_llm_result = None   # 直接LLM分支输出（开关关闭时保持 None）

    while current_round <= MAX_ROUNDS:

        # ── Step 2: 知识检索 ─────────────────────────────────────────────────
        retrieval_result = retrieve_process(question, agent_a_result)
        user_ctx_text    = context if context and context.strip() else ""
        deduped          = _dedup_against_user_context(retrieval_result, user_ctx_text)
        all_evidence.extend(deduped)
        retrieval_history.append({
            "round":          current_round,
            "evidence_count": len(retrieval_result),
            "total_evidence": len(all_evidence)
        })

        # ── Step 3: Agent B ──────────────────────────────────────────────────
        agent_b_result = agent_b.run(question, all_evidence)

        # ── 直接LLM分支（仅第1轮，开关控制）────────────────────────────────
        if enable_direct_branch and agent_direct_llm is not None and current_round == 1:
            direct_llm_result = agent_direct_llm.run(
                question=question,
                agent_b_result=agent_b_result,
                task_mode=task_mode,
                verbose=False
            )

        # ── Step 4: Agent C ──────────────────────────────────────────────────
        contextual_question = f"原问题{question}\n当前识别框架为{fod}。"
        question_pico       = agent_a_result.get("extraction", {}).get("elements", {})
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
                "final_decision": {"decision": "INSUFFICIENT_EVIDENCE",
                                   "confidence": 0.0, "reason": "没有足够BPA"},
                "fusion_result": {}, "belief_plausibility": {}
            }
        else:
            agent_d_result = agent_d.run(question, fod, bpa_list)

        # 记录本轮到 Agent E 的推理历史（与 main.py 完全一致）
        if bpa_list:
            all_opts = set()
            for b in bpa_list:
                all_opts.update(k for k in b.keys() if k != 'uncertainty_theta')
            bpa_summary = {
                "option_avg_masses": {
                    opt: round(sum(b.get(opt, 0) for b in bpa_list) / len(bpa_list), 4)
                    for opt in sorted(all_opts)
                },
                "avg_uncertainty": round(
                    sum(b.get('uncertainty_theta', 0) for b in bpa_list) / len(bpa_list), 4
                ),
                "bpa_count": len(bpa_list)
            }
        else:
            bpa_summary = {"bpa_count": 0}
        agent_e.add_reasoning_round(
            round_num=current_round,
            evidence_count=len(retrieval_result),
            bpa_summary=bpa_summary,
            note=f"第{current_round}轮推理完成"
        )

        # 完备性分析（与 main.py 完全一致）
        raw_fused  = agent_d_result.get("fusion_result", {}).get("fused_bpa", {})
        opt_masses = {k: v for k, v in raw_fused.items() if k != 'uncertainty_theta'}
        top_mass   = max(opt_masses.values()) if opt_masses else 0.0
        u_mass     = raw_fused.get('uncertainty_theta', 1.0)
        fused_bpa_cc = {
            'support_hypothesis': top_mass,
            'against_hypothesis': round(sum(opt_masses.values()) - top_mass, 4),
            'uncertainty': u_mass
        }
        belief_pl_cc = {
            'hypothesis_positive': {
                'belief':               top_mass,
                'plausibility':         round(top_mass + u_mass, 4),
                'uncertainty_interval': u_mass
            },
            'hypothesis_negative': {
                'belief':               round(max(0.0, 1.0 - top_mass - u_mass), 4),
                'plausibility':         round(1.0 - top_mass, 4),
                'uncertainty_interval': u_mass
            }
        }
        conflict_coef = agent_d_result.get("fusion_result", {}).get("conflict_coefficient", 0)
        completeness  = controller.analyze_completeness(fused_bpa_cc, belief_pl_cc, conflict_coef)

        if not completeness["should_continue"] or current_round >= MAX_ROUNDS:
            break
        current_round += 1

    # ── Step 6: Agent E (DS 分支) ────────────────────────────────────────────
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
            ev_data      = ev.get("evaluation", {})
            rich_content = ev_data.get("content_for_generator") or \
                           f"[Raw Snippet]: {ev_data.get('processed_input_snippet', 'N/A')}"
            enhanced.append({
                "source":   ev.get("source_type", "Unknown"),
                "content":  rich_content,
                "score":    max(
                    (v for k, v in ev_data.get("bpa_distribution", {}).items()
                     if k != 'uncertainty_theta'),
                    default=0.0
                ),
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

    # ── 最终聚合（开关控制）──────────────────────────────────────────────────
    final_aggregated_result = None
    if enable_direct_branch and agent_final_agg is not None and direct_llm_result is not None:
        final_aggregated_result = agent_final_agg.run(
            question=question,
            ds_result=agent_e_result,
            direct_llm_result=direct_llm_result,
            task_mode=task_mode,
            verbose=False
        )

    return {
        "agent_e_result":          agent_e_result,
        "direct_llm_result":       direct_llm_result,
        "final_aggregated_result": final_aggregated_result,
        "task_mode":               task_mode,
        # 附加调试信息（可选保留）
        "retrieval_history":   retrieval_history,
        "total_evidence_count": len(enhanced),
        "agent_a_analysis":    agent_a_result,
        "agent_d_fusion":      agent_d_result,
    }


# ----------------------------------------------------------
# 批量评估器
# ----------------------------------------------------------
class PubMedQAEvaluatorV2:
    """
    支持三路答案评估：
      1. DS 分支（Agent E 输出）
      2. 直接 LLM 分支（AgentDirectLLM 输出）
      3. 最终聚合（AgentFinalAggregator 输出）

    主要评估指标取自"最终聚合"（当分支开启时）或"DS 分支"（当分支关闭时）。
    """

    def __init__(self):
        self.results = []
        # 三路计数器
        self.counts = {
            "ds":         {"total": 0, "correct": 0},
            "direct_llm": {"total": 0, "correct": 0},
            "aggregated": {"total": 0, "correct": 0},
        }

        # A/B/C/D 实例复用（无跨条目状态），AgentE 每条重建
        self.agent_a    = AgentA()
        self.agent_b    = AgentB()
        self.agent_c    = AgentC()
        self.agent_d    = AgentD()
        self.controller = CompletenessController()

        # 根据开关决定是否初始化新增智能体
        if ENABLE_DIRECT_LLM_BRANCH:
            self.agent_direct_llm = AgentDirectLLM()
            self.agent_final_agg  = AgentFinalAggregator()
        else:
            self.agent_direct_llm = None
            self.agent_final_agg  = None

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

            pipeline_result = run_pipeline_single(
                question=question,
                context=context,
                agent_a=self.agent_a,
                agent_b=self.agent_b,
                agent_c=self.agent_c,
                agent_d=self.agent_d,
                agent_e=agent_e,
                controller=self.controller,
                agent_direct_llm=self.agent_direct_llm,
                agent_final_agg=self.agent_final_agg,
                enable_direct_branch=ENABLE_DIRECT_LLM_BRANCH,
            )

            agent_e_result          = pipeline_result.get("agent_e_result", {})
            direct_llm_result       = pipeline_result.get("direct_llm_result")
            final_aggregated_result = pipeline_result.get("final_aggregated_result")

            gt_norm = normalize_answer(ground_truth)

            # 三路答案提取
            pred_ds    = extract_ds_answer(agent_e_result)
            pred_llm   = extract_direct_llm_answer(direct_llm_result)
            pred_agg   = extract_aggregated_answer(final_aggregated_result)

            # 主要预测：有聚合结果用聚合，否则用 DS
            pred_main = pred_agg if (ENABLE_DIRECT_LLM_BRANCH and pred_agg) else pred_ds

            return {
                "pmid":          pmid,
                "question":      question,
                "ground_truth":  ground_truth,
                # 主要评估字段
                "predicted":          pred_main,
                "is_correct":         (pred_main == gt_norm),
                # 三路详细字段
                "pred_ds":            pred_ds,
                "pred_direct_llm":    pred_llm,
                "pred_aggregated":    pred_agg,
                "correct_ds":         (pred_ds  == gt_norm),
                "correct_direct_llm": (pred_llm == gt_norm),
                "correct_aggregated": (pred_agg == gt_norm),
                # 置信度
                "confidence_ds":          agent_e_result.get("confidence_score"),
                "confidence_direct_llm":  direct_llm_result.get("confidence_score") if direct_llm_result else None,
                "confidence_aggregated":  final_aggregated_result.get("confidence_score") if final_aggregated_result else None,
                # 推理摘要
                "reasoning_ds":          agent_e_result.get("reasoning", ""),
                "reasoning_direct_llm":  direct_llm_result.get("reasoning", "") if direct_llm_result else "",
                "reasoning_aggregated":  final_aggregated_result.get("reasoning", "") if final_aggregated_result else "",
                # 聚合说明
                "integration_note": final_aggregated_result.get("integration_note", "") if final_aggregated_result else "",
                "agreement":        final_aggregated_result.get("agreement", "") if final_aggregated_result else "",
                # 原始结果（完整保留以备离线分析）
                "raw_agent_e_result":          agent_e_result,
                "raw_direct_llm_result":       direct_llm_result,
                "raw_final_aggregated_result": final_aggregated_result,
                # 调试信息
                "task_mode":         pipeline_result.get("task_mode"),
                "retrieval_history": pipeline_result.get("retrieval_history"),
                "error":             None,
            }

        except Exception:
            return {
                "pmid":               pmid,
                "question":           question,
                "ground_truth":       ground_truth,
                "predicted":          None,
                "is_correct":         False,
                "pred_ds":            None,
                "pred_direct_llm":    None,
                "pred_aggregated":    None,
                "correct_ds":         False,
                "correct_direct_llm": False,
                "correct_aggregated": False,
                "confidence_ds":          None,
                "confidence_direct_llm":  None,
                "confidence_aggregated":  None,
                "reasoning_ds":           "",
                "reasoning_direct_llm":   "",
                "reasoning_aggregated":   "",
                "integration_note": "",
                "agreement":        "",
                "raw_agent_e_result":          None,
                "raw_direct_llm_result":       None,
                "raw_final_aggregated_result": None,
                "task_mode":         None,
                "retrieval_history": [],
                "error":             traceback.format_exc(),
            }

    # ── 批量评估 ─────────────────────────────────────────────────────────────
    def run(self):
        print("=" * 80)
        print("PubMedQA 批量评估 v2  —  含直接LLM分支 + 最终聚合")
        print(f"直接LLM分支: {'已开启' if ENABLE_DIRECT_LLM_BRANCH else '已关闭'}")
        print("=" * 80)

        data = self.load_data()

        for i, item in enumerate(tqdm(data, desc="评估进度")):
            rec = self.run_single(item)
            self.results.append(rec)

            gt_norm = normalize_answer(rec["ground_truth"])

            # 更新三路计数器
            for key, pred_key, correct_key in [
                ("ds",         "pred_ds",         "correct_ds"),
                ("direct_llm", "pred_direct_llm", "correct_direct_llm"),
                ("aggregated", "pred_aggregated",  "correct_aggregated"),
            ]:
                if rec[pred_key] is not None and rec[pred_key] != "":
                    self.counts[key]["total"]   += 1
                    if rec[correct_key]:
                        self.counts[key]["correct"] += 1

            # tqdm 行打印
            flag   = "✓" if rec["is_correct"] else "✗"
            gt_s   = str(rec["ground_truth"] or "")
            main_s = str(rec["predicted"]    or "N/A")
            ds_s   = str(rec["pred_ds"]      or "N/A")
            llm_s  = str(rec["pred_direct_llm"] or "-") if ENABLE_DIRECT_LLM_BRANCH else "-"
            agg_s  = str(rec["pred_aggregated"]  or "-") if ENABLE_DIRECT_LLM_BRANCH else "-"
            err_s  = (" ERR: " + rec["error"].splitlines()[0][:60]) if rec["error"] else ""

            # 实时准确率：以"聚合"为主，关闭分支则以 DS 为主
            main_cnt = self.counts["aggregated"] if ENABLE_DIRECT_LLM_BRANCH else self.counts["ds"]
            acc_main = main_cnt["correct"] / main_cnt["total"] * 100 if main_cnt["total"] else 0

            tqdm.write(
                f"[{i+1:>5}/{len(data)}] {flag}  PMID={rec['pmid']:<10}"
                f"  GT={gt_s:<5}  Main={main_s:<5}  DS={ds_s:<5}"
                f"  LLM={llm_s:<5}  Agg={agg_s:<5}  Acc={acc_main:.1f}%{err_s}"
            )

            if (i + 1) % SAVE_INTERVAL == 0:
                self._save(interim=True)

        self._save(interim=False)
        self._print_summary()

    # ── 保存结果 ─────────────────────────────────────────────────────────────
    def _save(self, interim: bool):
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "interim_" if interim else "final_"
        fpath  = f"{OUTPUT_DIR}/pubmedqa_v2_{prefix}{ts}.json"

        def _acc(key):
            c = self.counts[key]
            return round(c["correct"] / c["total"] * 100, 4) if c["total"] else 0

        out = {
            "meta": {
                "timestamp":               ts,
                "data_path":               DATA_PATH,
                "test_limit":              TEST_LIMIT,
                "max_rounds":              MAX_ROUNDS,
                "enable_direct_llm_branch": ENABLE_DIRECT_LLM_BRANCH,
                "agent_c_max_tokens":      AGENT_C_MAX_TOKENS,
                "total_count":             len(self.results),
                # 三路准确率
                "accuracy_ds":            _acc("ds"),
                "accuracy_direct_llm":    _acc("direct_llm"),
                "accuracy_aggregated":    _acc("aggregated"),
            },
            "results": [
                # 保存时去掉体积较大的 raw_* 字段（可按需保留）
                {k: v for k, v in r.items()
                 if k not in ("raw_agent_e_result",
                               "raw_direct_llm_result",
                               "raw_final_aggregated_result")}
                for r in self.results
            ]
        }
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        tqdm.write(f"\n  → 已保存：{fpath}\n")

    # ── 摘要统计 ─────────────────────────────────────────────────────────────
    def _print_summary(self):
        print("\n" + "=" * 80)
        print("评估摘要  —  MEDAR-QA PubMedQA v2")
        print("=" * 80)

        total = len(self.results)
        print(f"总条数   : {total}")

        branches = [
            ("DS 分支（Agent E）",        "ds",         "pred_ds",         "correct_ds"),
        ]
        if ENABLE_DIRECT_LLM_BRANCH:
            branches += [
                ("直接LLM分支",             "direct_llm", "pred_direct_llm", "correct_direct_llm"),
                ("最终聚合（主评估指标）",   "aggregated", "pred_aggregated",  "correct_aggregated"),
            ]

        for label, key, pred_key, correct_key in branches:
            c = self.counts[key]
            t, co = c["total"], c["correct"]
            acc_str = f"{co/t*100:.2f}%" if t else "N/A"
            print(f"\n  [{label}]")
            print(f"    有效预测数 : {t}")
            print(f"    正确数     : {co}")
            print(f"    准确率     : {acc_str}")

            print(f"    按标准答案分类：")
            for lbl in ("yes", "no", "maybe"):
                subset = [r for r in self.results if normalize_answer(r["ground_truth"]) == lbl]
                correct_subset = [r for r in subset if r[correct_key]]
                if subset:
                    print(f"      {lbl.capitalize():<6}: {len(correct_subset)}/{len(subset)}"
                          f"  ({len(correct_subset)/len(subset)*100:.1f}%)")

            print(f"    预测分布：")
            dist = Counter(str(r.get(pred_key) or "(空)") for r in self.results)
            for k, v in dist.most_common():
                print(f"      {k:<12}: {v}")

        # 聚合一致性统计（仅当分支开启时）
        if ENABLE_DIRECT_LLM_BRANCH:
            agree_count    = sum(1 for r in self.results if r.get("agreement") == "agree")
            disagree_count = sum(1 for r in self.results
                                 if r.get("agreement") == "disagree")
            valid_agg      = sum(1 for r in self.results if r.get("agreement") in ("agree", "disagree"))
            print(f"\n  [两分支一致性]")
            print(f"    一致 (agree)   : {agree_count}/{valid_agg}")
            print(f"    不一致 (disagree): {disagree_count}/{valid_agg}")

        # 错误统计
        err_n = sum(1 for r in self.results if r.get("error"))
        if err_n:
            print(f"\n运行时异常数量 : {err_n}")
        print("=" * 80)


if __name__ == "__main__":
    evaluator = PubMedQAEvaluatorV2()
    evaluator.run()
