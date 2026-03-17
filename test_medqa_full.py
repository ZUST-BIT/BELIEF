"""
MEDAR-QA MedQA 批量测试脚本（完整管线版）
============================================
完全复刻 main.py 的最新推理流程，并适配 MedQA 四选一（A/B/C/D）格式：
  - 直接LLM推理分支（ENABLE_DIRECT_LLM_BRANCH 开关控制）
  - 最终聚合智能体（综合 DS 与 LLM 两条路径）
  - 三路答案同时评估（DS 分支 / 直接LLM 分支 / 最终聚合）

数据格式：MedQA JSONL，每行：
  {realidx, question, options:{A:...,B:...,C:...,D:...},
   answer, answer_idx, meta_info, metamap_phrases}

答案提取优先级（MCQ，主评估指标在前）：
  1. 聚合结果  — final_aggregated_result["final_answer"]（AgentFinalAggregator，需开启分支）
  2. 直接LLM   — direct_llm_result["selected_option"]
  3. DS 分支   — agent_e_result["selected_option"]（AgentE 使用 Prompt_E_Test_MCQ）
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
DATA_PATH     = "data/medqa_sample.jsonl"
OUTPUT_DIR    = "TEST_RESULTS/medqa_full"
TEST_LIMIT    = 100         # None = 全部；整数 = 只跑前 N 条
MAX_ROUNDS    = 1            # 与 main.py 保持一致
SAVE_INTERVAL = 10           # 每隔多少条保存一次中间结果

# MCQ 必须开启 DirectLLM 分支，AgentFinalAggregator 负责输出 A/B/C/D
ENABLE_DIRECT_LLM_BRANCH: bool = True

# Qwen3 等思考模型专用：调高 max_tokens 防止 think 块耗尽 token 限额
AGENT_C_MAX_TOKENS = 12000   # think 块 ~5000 + JSON 输出 ~1000，留足余量
AGENT_E_MAX_TOKENS = 4096    # AgentE 生成报告
# ==================================================


# ----------------------------------------------------------
# 补丁函数定义（在 import agents 之后调用 _apply_patches()）
# ----------------------------------------------------------
def _patched_remove_think_tags(self, text: str) -> str:
    """
    增强版 _remove_think_tags：
    当 </think> 后无内容（max_tokens 被 think 块耗尽）时，
    回退到 think 块内部用状态机提取最大合法 JSON 对象，
    或直接返回 think 块内容供下游解析器（MCQ/反射标签）提取答案。
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
                if esc:                    esc = False
                elif ch == '\\' and in_str: esc = True
                elif ch == '"':            in_str = not in_str
                elif not in_str:
                    if ch == '{':          depth += 1
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
        # </think> 后为空 → 优先提取 JSON，否则返回 think 内容供下游解析
        print("⚠️  think 块耗尽 max_tokens，尝试从 think 内容中提取 JSON...")
        think_body = re.sub(r'</?think>', '', text).strip()
        found = _scan_largest_json(think_body)
        if found:
            return found
        # 无 JSON，直接返回推理内容（MCQ 解析器可从中提取 "Answer: X" 等模式）
        return think_body

    # 情况2: 只有 <think>（被截断，没有 </think>）
    if '<think>' in text:
        before = text.split('<think>')[0].strip()
        if before:
            return before
        think_body = text.split('<think>', 1)[1]
        found = _scan_largest_json(think_body)
        return found if found else think_body.strip()

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
def normalize_mcq_answer(text: str) -> str:
    """将各种格式的答案规范化为单个大写字母 A / B / C / D，失败返回空字符串"""
    if not text:
        return ""
    t = str(text).strip().upper()
    # 直接是单字母
    if t in ("A", "B", "C", "D"):
        return t
    # 以选项字母开头（如 "A: xxx" 或 "A."）
    if t and t[0] in "ABCD":
        return t[0]
    # 扫描文本中最后出现的选项字母（比最先出现更可靠，模型通常在结尾给出结论）
    for ch in reversed(t):
        if ch in "ABCD":
            return ch
    return ""


def extract_aggregated_answer(result: Optional[Dict]) -> str:
    """从 AgentFinalAggregator 结果中提取 MCQ 答案（final_answer 字段）"""
    if not result:
        return ""
    raw = result.get("final_answer", "")
    ans = normalize_mcq_answer(raw)
    if ans:
        return ans
    # 降级：从 reasoning 文本中最后出现的 A/B/C/D
    reasoning = str(result.get("reasoning", "")).upper()
    for ch in reversed(reasoning):
        if ch in "ABCD":
            return ch
    return ""


def extract_ds_branch_answer(result: Optional[Dict]) -> str:
    """从 AgentE（DS 分支，MCQ Prompt）结果中提取 selected_option"""
    if not result:
        return ""
    raw = result.get("selected_option", "")
    ans = normalize_mcq_answer(raw)
    if ans:
        return ans
    # 降级：reasoning 文本
    reasoning = str(result.get("reasoning", "")).upper()
    for ch in reversed(reasoning):
        if ch in "ABCD":
            return ch
    return ""


def extract_direct_llm_answer(result: Optional[Dict]) -> str:
    """从 AgentDirectLLM（MCQ 模式）结果中提取 selected_option"""
    if not result:
        return ""
    raw = result.get("selected_option", "")
    ans = normalize_mcq_answer(raw)
    if ans:
        return ans
    # 降级：reasoning 文本
    reasoning = str(result.get("reasoning", "")).upper()
    for ch in reversed(reasoning):
        if ch in "ABCD":
            return ch
    return ""


def format_question_with_options(question: str, options: Dict[str, str]) -> str:
    """将问题正文与选项拼接为 AgentA 可读格式"""
    opts = "\n".join(f"{k}: {v}" for k, v in sorted(options.items()))
    return f"{question}\n\nOptions:\n{opts}"


# ----------------------------------------------------------
# 核心流程（完全复刻 main.py，不含 user context，增加 MCQ 选项注入）
# ----------------------------------------------------------
def run_pipeline_single(
        question: str,
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
    question 参数已包含选项文本（由调用方通过 format_question_with_options 拼接）。

    Returns:
        {
            "agent_e_result":           ...,   # DS 分支输出（AgentE 报告，MCQ 时格式不准）
            "direct_llm_result":        ...,   # 直接LLM分支输出（MCQ 主要答案来源）
            "final_aggregated_result":  ...,   # 最终聚合输出（MCQ 最终答案字段 final_answer）
            "task_mode":                ...,   # SELECTION / OPEN_REASONING
            "retrieval_history":        ...,
            "total_evidence_count":     ...,
            "agent_a_analysis":         ...,
            "agent_d_fusion":           ...,
        }
    """
    # ── Step 1: Agent A 问题分析 ────────────────────────────────────────────
    agent_a_result = agent_a.run(question)
    fod            = agent_a_result.get("frame_of_discernment", ["H", "¬H"])
    task_mode      = agent_a_result.get("task_mode", "SELECTION")

    # 多轮检索初始化
    current_round     = 1
    all_evidence      = []
    retrieval_history = []

    # 各轮结果占位（防止循环未执行时引用未定义变量）
    agent_b_result = {}
    agent_c_result = {}
    agent_d_result = {
        "final_decision": {"decision": "INSUFFICIENT_EVIDENCE",
                           "confidence": 0.0, "reason": "没有足够的有效证据"},
        "fusion_result": {}, "belief_plausibility": {}
    }
    direct_llm_result = None

    while current_round <= MAX_ROUNDS:

        # ── Step 2: 知识检索 ──────────────────────────────────────────────
        retrieval_result = retrieve_process(question, agent_a_result)
        all_evidence.extend(retrieval_result)
        retrieval_history.append({
            "round":          current_round,
            "evidence_count": len(retrieval_result),
            "total_evidence": len(all_evidence)
        })

        # ── Step 3: Agent B PICO 提取 ─────────────────────────────────────
        agent_b_result = agent_b.run(question, all_evidence)

        # ── 直接LLM分支（仅第1轮，MCQ 必须开启）────────────────────────────
        if enable_direct_branch and agent_direct_llm is not None and current_round == 1:
            direct_llm_result = agent_direct_llm.run(
                question=question,
                agent_b_result=agent_b_result,
                task_mode=task_mode,    # AgentA 对 MCQ 应输出 "SELECTION"
                verbose=False
            )

        # ── Step 4: Agent C 证据评估与 BPA 计算 ───────────────────────────
        contextual_question = f"原问题{question}\n当前识别框架为{fod}。"
        question_pico       = agent_a_result.get("extraction", {}).get("elements", {})
        agent_c_result = agent_c.run(
            hypothesis=contextual_question,
            agent_b_result=agent_b_result,
            question_pico=question_pico,
            frame_of_discernment=fod,
            verbose=False
        )

        # ── Step 5: Agent D 多证据融合 ────────────────────────────────────
        bpa_list = agent_c_result.get("bpa_list", [])
        if not bpa_list:
            agent_d_result = {
                "final_decision": {"decision": "INSUFFICIENT_EVIDENCE",
                                   "confidence": 0.0, "reason": "没有足够BPA"},
                "fusion_result": {}, "belief_plausibility": {}
            }
        else:
            agent_d_result = agent_d.run(question, fod, bpa_list)

        # 记录本轮推理历史到 Agent E（与 main.py 完全一致）
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

    # ── Step 6: Agent E（DS 分支，MCQ 格式不准），仍运行以保持与 main.py 一致 ──
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

    # ── 最终聚合（MCQ 必须开启，输出 final_answer 字段含 A/B/C/D）────────────
    final_aggregated_result = None
    if enable_direct_branch and agent_final_agg is not None and direct_llm_result is not None:
        final_aggregated_result = agent_final_agg.run(
            question=question,
            ds_result=agent_e_result,
            direct_llm_result=direct_llm_result,
            task_mode=task_mode,   # "SELECTION" → Prompt_FinalAggregator（MCQ 专用）
            verbose=False
        )

    return {
        "agent_e_result":          agent_e_result,
        "direct_llm_result":       direct_llm_result,
        "final_aggregated_result": final_aggregated_result,
        "task_mode":               task_mode,
        "retrieval_history":       retrieval_history,
        "total_evidence_count":    len(enhanced),
        "agent_a_analysis":        agent_a_result,
        "agent_d_fusion":          agent_d_result,
    }


# ----------------------------------------------------------
# 批量评估器
# ----------------------------------------------------------
class MedQAEvaluator:
    """
    使用 MEDAR-QA 完整多智能体管线在 MedQA 数据集上进行批量评估。
    支持三路答案同时评估：
      1. DS 分支（AgentE 输出，MCQ 时格式不准，仅作参考）
      2. 直接LLM 分支（AgentDirectLLM SELECTION 模式，selected_option）
      3. 最终聚合（AgentFinalAggregator，final_answer，MCQ 主评估指标）
    """

    def __init__(self):
        if not ENABLE_DIRECT_LLM_BRANCH:
            print("⚠️  ENABLE_DIRECT_LLM_BRANCH=False：仅运行 DS 分支（AgentE MCQ Prompt）。"
                  "建议开启以获得更高准确率（聚合分支综合两路推理）。")

        # A/B/C/D 实例复用；AgentE 每条重建（隔离推理历史）
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

        self.results: List[Dict] = []
        self.counts = {
            "ds":         {"total": 0, "correct": 0},
            "direct_llm": {"total": 0, "correct": 0},
            "aggregated": {"total": 0, "correct": 0},
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 数据加载 ────────────────────────────────────────────────────────────
    def load_data(self) -> List[Dict]:
        data = []
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if TEST_LIMIT is not None and i >= TEST_LIMIT:
                    break
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        print(f"共加载 {len(data)} 条 MedQA 测试数据")
        return data

    @staticmethod
    def normalize_gt(item: Dict) -> str:
        """从数据条目提取并规范化标准答案字母"""
        # 优先使用 answer_idx（直接是字母，如 "B"）
        raw = item.get("answer_idx", "") or ""
        ans = normalize_mcq_answer(raw)
        if ans:
            return ans
        # 降级：用 answer 完整文本匹配 options，找到对应字母
        answer_text = (item.get("answer", "") or "").strip().lower()
        for k, v in (item.get("options", {}) or {}).items():
            if v.strip().lower() == answer_text:
                return k.upper()
        return ""

    # ── 单条测试 ────────────────────────────────────────────────────────────
    def run_single(self, item: Dict) -> Dict:
        question_text = item.get("question", "")
        options       = item.get("options", {})
        ground_truth  = self.normalize_gt(item)
        realidx       = item.get("realidx")
        meta_info     = item.get("meta_info", "")

        # 将选项注入问题文本，AgentA 据此构建 FoD = 选项文本列表
        full_question = format_question_with_options(question_text, options)

        try:
            # AgentE 每条重建，防止跨题历史污染
            agent_e = AgentE()

            pipeline_result = run_pipeline_single(
                question=full_question,
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

            # 三路答案提取
            pred_ds  = extract_ds_branch_answer(agent_e_result)
            pred_llm = extract_direct_llm_answer(direct_llm_result)
            pred_agg = extract_aggregated_answer(final_aggregated_result)

            # 主要预测：聚合结果优先，次为 DirectLLM，最后 DS
            pred_main = pred_agg or pred_llm or pred_ds

            return {
                "realidx":     realidx,
                "meta_info":   meta_info,
                "question":      question_text,
                "options":       options,
                "ground_truth":  ground_truth,
                # 主要评估字段
                "predicted":     pred_main,
                "is_correct":    (pred_main == ground_truth) and bool(pred_main),
                # 三路详细字段
                "pred_ds":            pred_ds,
                "pred_direct_llm":    pred_llm,
                "pred_aggregated":    pred_agg,
                "correct_ds":         (pred_ds  == ground_truth) and bool(pred_ds),
                "correct_direct_llm": (pred_llm == ground_truth) and bool(pred_llm),
                "correct_aggregated": (pred_agg == ground_truth) and bool(pred_agg),
                # 置信度
                "confidence_ds":          agent_e_result.get("confidence_score"),
                "confidence_direct_llm":  (direct_llm_result or {}).get("confidence_score"),
                "confidence_aggregated":  (final_aggregated_result or {}).get("confidence_score"),
                # 推理摘要
                "reasoning_ds":          agent_e_result.get("reasoning", ""),
                "reasoning_direct_llm":  (direct_llm_result or {}).get("reasoning", ""),
                "reasoning_aggregated":  (final_aggregated_result or {}).get("reasoning", ""),
                # 聚合元信息
                "integration_note": (final_aggregated_result or {}).get("integration_note", ""),
                "agreement":        (final_aggregated_result or {}).get("agreement", ""),
                # 调试信息
                "task_mode":         pipeline_result.get("task_mode"),
                "fod":               pipeline_result.get("agent_a_analysis", {})
                                        .get("frame_of_discernment", []),
                "retrieval_history": pipeline_result.get("retrieval_history"),
                "evidence_count":    pipeline_result.get("total_evidence_count", 0),
                # 原始结果（完整保留以备离线分析）
                "raw_agent_e_result":          agent_e_result,
                "raw_direct_llm_result":       direct_llm_result,
                "raw_final_aggregated_result": final_aggregated_result,
                "error": None,
            }

        except Exception:
            return {
                "realidx":     realidx,
                "meta_info":   meta_info,
                "question":      question_text,
                "options":       options,
                "ground_truth":  ground_truth,
                "predicted":     None,
                "is_correct":    False,
                "pred_ds":            None,
                "pred_direct_llm":    None,
                "pred_aggregated":    None,
                "correct_ds":         False,
                "correct_direct_llm": False,
                "correct_aggregated": False,
                "confidence_ds":           None,
                "confidence_direct_llm":   None,
                "confidence_aggregated":   None,
                "reasoning_ds":           "",
                "reasoning_direct_llm":   "",
                "reasoning_aggregated":   "",
                "integration_note": "",
                "agreement":        "",
                "task_mode":      None,
                "fod":            [],
                "retrieval_history": [],
                "evidence_count": 0,
                "raw_agent_e_result":          None,
                "raw_direct_llm_result":       None,
                "raw_final_aggregated_result": None,
                "error": traceback.format_exc(),
            }

    # ── 批量评估 ────────────────────────────────────────────────────────────
    def run(self):
        print("=" * 80)
        print("MedQA 批量评估  —  MEDAR-QA 完整管线（MCQ A/B/C/D）")
        print(f"直接LLM分支: {'已开启' if ENABLE_DIRECT_LLM_BRANCH else '已关闭'}")
        print(f"数据路径   : {DATA_PATH}")
        print(f"输出目录   : {OUTPUT_DIR}")
        print(f"最大轮次   : {MAX_ROUNDS}")
        print(f"AgentC max_tokens: {AGENT_C_MAX_TOKENS}")
        print("=" * 80)

        data = self.load_data()

        for i, item in enumerate(tqdm(data, desc="MedQA 评估进度")):
            rec = self.run_single(item)
            self.results.append(rec)

            gt = str(rec["ground_truth"] or "")

            # 更新三路计数器（仅计有效预测）
            for key, pred_key, correct_key in [
                ("ds",         "pred_ds",         "correct_ds"),
                ("direct_llm", "pred_direct_llm", "correct_direct_llm"),
                ("aggregated", "pred_aggregated",  "correct_aggregated"),
            ]:
                if rec[pred_key]:
                    self.counts[key]["total"]   += 1
                    if rec[correct_key]:
                        self.counts[key]["correct"] += 1

            # 主路实时准确率：聚合优先，分支关闭时用 DS
            main_cnt = self.counts["aggregated"] if ENABLE_DIRECT_LLM_BRANCH else self.counts["ds"]
            acc = main_cnt["correct"] / main_cnt["total"] * 100 if main_cnt["total"] else 0.0

            flag    = "✓" if rec["is_correct"] else "✗"
            ridx_s  = str(rec.get("realidx", i))
            main_s  = str(rec["predicted"]       or "N/A")
            ds_s    = str(rec["pred_ds"]         or "-")
            llm_s   = str(rec["pred_direct_llm"] or "-")
            agg_s   = str(rec["pred_aggregated"] or "-")
            err_s   = (" ERR: " + rec["error"].splitlines()[0][:60]) if rec["error"] else ""

            tqdm.write(
                f"[{i+1:>5}/{len(data)}] {flag}  idx={ridx_s:<5}  GT={gt:<2}"
                f"  Main={main_s:<2}  DS={ds_s:<2}  LLM={llm_s:<2}  Agg={agg_s:<2}"
                f"  Acc={acc:.1f}%{err_s}"
            )

            if (i + 1) % SAVE_INTERVAL == 0:
                self._save(interim=True)

        self._save(interim=False)
        self._print_summary()

    # ── 保存结果 ────────────────────────────────────────────────────────────
    def _save(self, interim: bool):
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "interim_" if interim else "final_"
        fpath  = f"{OUTPUT_DIR}/medqa_{prefix}{ts}.json"

        def _acc(key):
            c = self.counts[key]
            return round(c["correct"] / c["total"] * 100, 4) if c["total"] else 0.0

        out = {
            "meta": {
                "timestamp":               ts,
                "data_path":               DATA_PATH,
                "test_limit":              TEST_LIMIT,
                "max_rounds":              MAX_ROUNDS,
                "enable_direct_llm_branch": ENABLE_DIRECT_LLM_BRANCH,
                "agent_c_max_tokens":      AGENT_C_MAX_TOKENS,
                "total_count":             len(self.results),
                "accuracy_ds":             _acc("ds"),
                "accuracy_direct_llm":     _acc("direct_llm"),
                "accuracy_aggregated":     _acc("aggregated"),
            },
            # 保存时去掉体积较大的 raw_* 字段（可按需保留）
            "results": [
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

    # ── 摘要统计 ────────────────────────────────────────────────────────────
    def _print_summary(self):
        print("\n" + "=" * 80)
        print("评估摘要  —  MEDAR-QA MedQA 批量测试")
        print("=" * 80)

        total = len(self.results)
        print(f"总题数   : {total}")

        branches = [
            ("DS 分支（AgentE，MCQ Prompt）",    "ds",
             "pred_ds",         "correct_ds"),
        ]
        if ENABLE_DIRECT_LLM_BRANCH:
            branches += [
                ("直接LLM分支（AgentDirectLLM）", "direct_llm",
                 "pred_direct_llm", "correct_direct_llm"),
                ("最终聚合（主评估指标）",          "aggregated",
                 "pred_aggregated",  "correct_aggregated"),
            ]

        for label, key, pred_key, correct_key in branches:
            c = self.counts[key]
            t, co = c["total"], c["correct"]
            acc_str = f"{co/t*100:.2f}%" if t else "N/A"
            print(f"\n  [{label}]")
            print(f"    有效预测数 : {t}")
            print(f"    正确数     : {co}")
            print(f"    准确率     : {acc_str}")

            # 按标准答案分类准确率
            print(f"    按标准答案分类：")
            for opt in ("A", "B", "C", "D"):
                subset = [r for r in self.results if r.get("ground_truth") == opt]
                if subset:
                    correct_subset = [r for r in subset if r.get(correct_key)]
                    print(f"      {opt}: {len(correct_subset)}/{len(subset)}"
                          f"  ({len(correct_subset)/len(subset)*100:.1f}%)")

            # 预测分布
            print(f"    预测分布：")
            dist = Counter(str(r.get(pred_key) or "(空)") for r in self.results)
            for k, v in dist.most_common():
                print(f"      {k:<12}: {v}")

        # 聚合一致性统计（仅当分支开启时）
        if ENABLE_DIRECT_LLM_BRANCH:
            agree_count    = sum(1 for r in self.results if r.get("agreement") == "agree")
            disagree_count = sum(1 for r in self.results if r.get("agreement") == "disagree")
            valid_agg      = sum(1 for r in self.results if r.get("agreement") in ("agree", "disagree"))
            print(f"\n  [两分支一致性]")
            print(f"    一致  (agree)   : {agree_count}/{valid_agg}")
            print(f"    不一致 (disagree): {disagree_count}/{valid_agg}")

        # 按 meta_info（step1 / step2&3）分类准确率
        print(f"\n  [按试题类型 (meta_info)]")
        meta_groups: Dict[str, Dict] = {}
        for r in self.results:
            mi = r.get("meta_info") or "unknown"
            if mi not in meta_groups:
                meta_groups[mi] = {"total": 0, "correct": 0}
            meta_groups[mi]["total"] += 1
            if r.get("is_correct"):
                meta_groups[mi]["correct"] += 1
        for mi, c in sorted(meta_groups.items()):
            acc_str = f"{c['correct']/c['total']*100:.1f}%" if c["total"] else "N/A"
            print(f"    {mi:<12}: {c['correct']}/{c['total']}  ({acc_str})")

        # 错误统计
        err_n = sum(1 for r in self.results if r.get("error"))
        if err_n:
            print(f"\n  运行时异常数量 : {err_n}")

        print("=" * 80)


# ----------------------------------------------------------
# 入口
# ----------------------------------------------------------
if __name__ == "__main__":
    evaluator = MedQAEvaluator()
    evaluator.run()
