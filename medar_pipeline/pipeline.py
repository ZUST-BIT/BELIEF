"""MEDAR-QA pipeline orchestration."""

import json
from datetime import datetime
from typing import Dict, Any, List

from agents import (
    QuestionAnalyzer,
    EvidenceAnalyzer,
    EvidenceEvaluator,
    EvidenceFusionEngine,
    ReportGenerator,
    EvidenceCompletenessController,
    DirectReasoningAgent,
    AnswerArbiter,
)
from retriever import retrieve_process
from .helpers import (
    JsonDict,
    print_title,
    print_step,
    dedup_against_user_context,
    build_user_context_evidence,
    build_bpa_summary,
    build_completeness_inputs,
    build_enhanced_evidence_input,
)


def run_pipeline(
    question: str,
    context: str,
    task_mode: str = "SELECTION",
    enable_direct_llm_branch: bool = True,
    max_rounds: int = 1,
) -> JsonDict:
    print_title("MEDAR-QA 医学证据推理系统")
    print(f"\n问题: {question}\n")
    if enable_direct_llm_branch:
        print("✔ 直接LLM分支开关：已开启（将并行D-S分支与直接LLM分支，最后聚合）")
    else:
        print("✓ 直接LLM分支开关：已关闭（仅运行现有D-S流程）")

    agent_a = QuestionAnalyzer()
    agent_b = EvidenceAnalyzer()
    agent_c = EvidenceEvaluator()
    agent_d = EvidenceFusionEngine()
    agent_e = ReportGenerator()
    controller = EvidenceCompletenessController()
    agent_direct_llm = DirectReasoningAgent() if enable_direct_llm_branch else None
    agent_final_agg = AnswerArbiter() if enable_direct_llm_branch else None
    direct_llm_result = None

    print_step("[步骤 1/6] QuestionAnalyzer：问题分析与实体提取")
    agent_a_result = agent_a.run(question)
    print("\n智能体A分析结果:")
    print(json.dumps(agent_a_result, ensure_ascii=False, indent=2))
    fod = agent_a_result.get("frame_of_discernment", ["H", "¬H"])

    current_round = 1
    all_evidence: List[JsonDict] = []
    retrieval_history: List[JsonDict] = []

    all_evidence.extend(build_user_context_evidence(context))

    while current_round <= max_rounds:
        print(f"\n{'='*80}")
        print(f"第 {current_round} 轮证据推理")
        print(f"{'='*80}")

        print_step(f"[轮次 {current_round} - 步骤 2/6] 知识检索：结合实体进行证据检索")
        retrieval_result = retrieve_process(question, agent_a_result)

        user_ctx_text = context if context and context.strip() else ""
        deduped = dedup_against_user_context(retrieval_result, user_ctx_text)
        skipped = len(retrieval_result) - len(deduped)
        if skipped:
            print(f"[去重] 检测到 {skipped} 条检索结果与用户上下文高度重叠，已过滤")
        all_evidence.extend(deduped)
        retrieval_history.append({
            "round": current_round,
            "evidence_count": len(retrieval_result),
            "total_evidence": len(all_evidence),
        })

        print_step(f"[轮次 {current_round} - 步骤 3/6] EvidenceAnalyzer：PICO提取与研究类型分类")
        agent_b_result = agent_b.run(question, all_evidence)
        print("\n智能体B分析结果:")
        print(json.dumps(agent_b_result, ensure_ascii=False, indent=2))

        if enable_direct_llm_branch and current_round == 1:
            print_step(f"[轮次 {current_round} - 直接LLM分支] 开始直接LLM推理")
            direct_llm_result = agent_direct_llm.run(
                question=question,
                evidence_result=agent_b_result,
                task_mode=task_mode,
                verbose=True,
            )
            print("\n直接LLM分支结果:")
            print(json.dumps(direct_llm_result, ensure_ascii=False, indent=2))

        print_step(f"[轮次 {current_round} - 步骤 4/6] EvidenceEvaluator：证据可靠性评估与BPA计算")
        contextual_question = f"原问题{question}\n当前识别框架为{fod}。"
        question_pico_data = agent_a_result.get("extraction", {}).get("elements", {})
        agent_c_result = agent_c.run(
            hypothesis=contextual_question,
            agent_b_result=agent_b_result,
            question_pico=question_pico_data,
            frame_of_discernment=fod,
            verbose=False,
        )
        print("\n智能体C评估结果:")
        print(json.dumps(agent_c_result, ensure_ascii=False, indent=2))

        print_step(f"[轮次 {current_round} - 步骤 5/6] EvidenceFusionEngine：多证据融合与决策")
        bpa_list = agent_c_result.get("bpa_list", [])
        if not bpa_list:
            print("⚠️ 没有有效的BPA，跳过融合步骤")
            agent_d_result = {
                "note": "No valid BPA for fusion",
                "final_decision": {
                    "decision": "INSUFFICIENT_EVIDENCE",
                    "confidence": 0.0,
                    "reason": "没有足够的有效证据进行推理",
                },
            }
        else:
            agent_d_result = agent_d.run(question, fod, bpa_list)
            print("\n智能体D融合结果:")
            print(json.dumps(agent_d_result, ensure_ascii=False, indent=2))

        bpa_summary = build_bpa_summary(bpa_list)
        agent_e.add_reasoning_round(
            round_num=current_round,
            evidence_count=len(retrieval_result),
            bpa_summary=bpa_summary,
            note=f"第{current_round}轮推理完成",
        )
        current_round += 1
        # print_step(f"[轮次 {current_round} - 完备性分析] 评估证据充分性")
        # completeness_inputs = build_completeness_inputs(agent_d_result)
        # completeness_result = controller.analyze_completeness(
        #     completeness_inputs["fused_bpa"],
        #     completeness_inputs["belief_pl"],
        #     completeness_inputs["conflict_coef"],
        # )

        # print(f"当前状态: {completeness_result['state']}")
        # print(f"Deng熵: {completeness_result.get('entropy', 0):.4f}")
        # print(f"决策建议: {completeness_result['action']}")
        # print(f"是否继续检索: {completeness_result['should_continue']}")
        # print(f"理由: {completeness_result['reason']}")
        # if "suggestion" in completeness_result:
        #     print(f"建议: {completeness_result['suggestion']}")

        # if not completeness_result["should_continue"]:
        #     print("\n✓ 证据充分性满足要求，终止检索")
        #     break

        # if current_round >= max_rounds:
        #     print(f"\n✓ 已达到最大轮次限制 ({max_rounds})，终止检索")
        #     break

        # print("\n→ 证据不充分，需要继续检索...")
        # current_round += 1

    print_title("[步骤 6/6] ReportGenerator：生成最终医学证据报告")

    agent_d_logic_note = f"""
    [Agent D Logic Trace]:
    - System Decision: {agent_d_result.get('final_decision', {}).get('decision')}
    - Evidence Mapping: Agent D identified specific evidence IDs as supporting the decision. 
      (Agent E MUST verify if this mapping aligns with clinical symptoms).
    """
    agent_e.add_reasoning_round(99, 0, {}, note=agent_d_logic_note)

    enhanced_evidence_input = build_enhanced_evidence_input(agent_c_result, all_evidence)
    final_decision = agent_d_result.get("final_decision", {
        "decision": "UNCERTAIN",
        "confidence": 0.0,
        "reason": "未获得决策结果",
    })

    fusion_result = agent_d_result.get("fusion_result", {})
    belief_analysis = agent_d_result.get("belief_plausibility", {})

    agent_e_result = agent_e.run(
        question=question,
        final_decision=final_decision,
        fusion_result=fusion_result,
        evidence_list=enhanced_evidence_input,
    )
    print("\n最终报告:")
    print(json.dumps(agent_e_result, ensure_ascii=False, indent=2))

    final_aggregated_result = None
    if enable_direct_llm_branch and direct_llm_result is not None:
        print_title("[最终聚合] 综合DS分支与直接LLM分支结果")

        ds_for_agg = {
            "answer": agent_e_result.get("answer", "maybe"),
            "confidence_score": agent_e_result.get("confidence_score", 0.0),
            "reasoning": agent_e_result.get("reasoning", ""),
            "source": "AgentE_FinalReport",
            "raw_d_decision": agent_d_result.get("final_decision", {}).get("decision"),
            "fusion_result": agent_d_result.get("fusion_result", {}),
        }
        print(f"[调试] 传入聚合器的 DS 数据: Ans={ds_for_agg['answer']}, Conf={ds_for_agg['confidence_score']}")

        final_aggregated_result = agent_final_agg.run(
            question=question,
            ds_result=ds_for_agg,
            direct_llm_result=direct_llm_result,
            task_mode=task_mode,
            verbose=True,
        )
        print("\n最终聚合结果:")
        print(json.dumps(final_aggregated_result, ensure_ascii=False, indent=2))

    final_result = {
        "question": question,
        "user_context": context.strip() if context and context.strip() else None,
        "has_user_context": bool(context and context.strip()),
        "total_rounds": current_round,
        "retrieval_history": retrieval_history,
        "total_evidence_count": len(enhanced_evidence_input),
        "agent_a_analysis": agent_a_result,
        "agent_b_analysis": agent_b_result,
        "agent_c_evaluation": agent_c_result,
        "agent_d_fusion": agent_d_result,
        "final_report": agent_e_result,
        "direct_llm_result": direct_llm_result,
        "final_aggregated_result": final_aggregated_result,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"TEST_RESULTS/test_demo_result_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("✓ 完整推理流程结束")
    print(f"✓ 总轮次: {current_round}")
    print(f"✓ 总证据数: {len(all_evidence)}")
    print(f"✓ 结果已保存到: {output_file}")
    print("=" * 80)

    return final_result
