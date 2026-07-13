"""Single-pass MEDAR-QA pipeline orchestration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from medar_agents import (
    AnswerArbiter,
    DirectReasoningAgent,
    EvidenceAnalyzer,
    EvidenceEvaluator,
    EvidenceFusionEngine,
    QuestionAnalyzer,
    ReportGenerator,
    normalize_task_mode,
)
from retriever import retrieve_process

from .helpers import (
    JsonDict,
    build_enhanced_evidence_input,
    build_user_context_evidence,
    dedup_against_user_context,
    print_step,
    print_title,
)


def run_pipeline(
    question: str,
    context: str = "",
    task_mode: str = "SELECTION",
    enable_direct_llm_branch: bool = True,
    *,
    output_dir: str | Path = "TEST_RESULTS",
) -> JsonDict:
    """Run one evidence-retrieval and reasoning pass for a medical question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    task_mode = normalize_task_mode(task_mode)

    print_title("MEDAR-QA 医学证据推理系统")
    print(f"\n问题: {question}\n")
    if enable_direct_llm_branch:
        print("✓ 双路径推理已开启：依次运行 D-S 证据路径和直接 LLM 路径，最后进行仲裁")
    else:
        print("✓ 直接 LLM 路径已关闭：仅运行 D-S 证据路径")

    question_analyzer = QuestionAnalyzer()
    evidence_analyzer = EvidenceAnalyzer()
    evidence_evaluator = EvidenceEvaluator()
    fusion_engine = EvidenceFusionEngine()
    report_generator = ReportGenerator()
    direct_reasoner = DirectReasoningAgent() if enable_direct_llm_branch else None
    answer_arbiter = AnswerArbiter() if enable_direct_llm_branch else None

    print_step("[步骤 1/6] QuestionAnalyzer：问题分析与实体提取")
    question_analysis = question_analyzer.run(question)
    print("\n问题分析结果:")
    print(json.dumps(question_analysis, ensure_ascii=False, indent=2))
    frame_of_discernment = question_analysis.get("frame_of_discernment", ["H", "¬H"])

    print_step("[步骤 2/6] 知识检索：根据问题分析结果检索证据")
    retrieved_evidence = retrieve_process(question, question_analysis) or []
    if not isinstance(retrieved_evidence, list):
        raise TypeError("retrieve_process must return a list of evidence records")

    user_context = context.strip() if isinstance(context, str) else ""
    unique_retrieved_evidence = dedup_against_user_context(retrieved_evidence, user_context)
    skipped_count = len(retrieved_evidence) - len(unique_retrieved_evidence)
    if skipped_count:
        print(f"[去重] {skipped_count} 条检索结果与用户上下文重复，已过滤")

    all_evidence: List[JsonDict] = build_user_context_evidence(user_context)
    all_evidence.extend(unique_retrieved_evidence)

    print_step("[步骤 3/6] EvidenceAnalyzer：PICO 提取与研究类型分类")
    evidence_analysis = evidence_analyzer.run(question, all_evidence)
    print("\n证据分析结果:")
    print(json.dumps(evidence_analysis, ensure_ascii=False, indent=2))

    direct_llm_result = None
    if direct_reasoner is not None:
        print_step("[直接 LLM 路径] 基于结构化证据进行独立推理")
        direct_llm_result = direct_reasoner.run(
            question=question,
            evidence_result=evidence_analysis,
            task_mode=task_mode,
            verbose=True,
        )
        print("\n直接 LLM 路径结果:")
        print(json.dumps(direct_llm_result, ensure_ascii=False, indent=2))

    print_step("[步骤 4/6] EvidenceEvaluator：证据可靠性评估与 BPA 计算")
    contextual_question = f"原问题：{question}\n当前识别框架：{frame_of_discernment}。"
    question_pico = question_analysis.get("extraction", {}).get("elements", {})
    evidence_evaluation = evidence_evaluator.run(
        hypothesis=contextual_question,
        agent_b_result=evidence_analysis,
        question_pico=question_pico,
        frame_of_discernment=frame_of_discernment,
        verbose=False,
    )
    print("\n证据评估结果:")
    print(json.dumps(evidence_evaluation, ensure_ascii=False, indent=2))

    print_step("[步骤 5/6] EvidenceFusionEngine：多证据融合与决策")
    bpa_list = evidence_evaluation.get("bpa_list", [])
    fusion_analysis = fusion_engine.run(question, frame_of_discernment, bpa_list)
    print("\n证据融合结果:")
    print(json.dumps(fusion_analysis, ensure_ascii=False, indent=2))

    print_title("[步骤 6/6] ReportGenerator：生成最终医学证据报告")
    enhanced_evidence = build_enhanced_evidence_input(evidence_evaluation, all_evidence)
    final_report = report_generator.run(
        question=question,
        final_decision=fusion_analysis.get("final_decision", {}),
        fusion_result=fusion_analysis.get("fusion_result", {}),
        evidence_list=enhanced_evidence,
        task_mode=task_mode,
    )
    print("\n最终报告:")
    print(json.dumps(final_report, ensure_ascii=False, indent=2))

    final_aggregated_result = None
    if answer_arbiter is not None and direct_llm_result is not None:
        print_title("[最终仲裁] 综合 D-S 证据路径与直接 LLM 路径")
        ds_result = {
            "answer": final_report.get("answer", "maybe" if task_mode == "YES_NO" else "UNKNOWN"),
            "confidence_score": final_report.get("confidence_score", 0.0),
            "reasoning": final_report.get("reasoning", ""),
            "source": "evidence_fusion_report",
            "raw_d_decision": fusion_analysis.get("final_decision", {}).get("decision"),
            "fusion_result": fusion_analysis.get("fusion_result", {}),
        }
        final_aggregated_result = answer_arbiter.run(
            question=question,
            ds_result=ds_result,
            direct_llm_result=direct_llm_result,
            task_mode=task_mode,
            verbose=True,
        )
        print("\n最终仲裁结果:")
        print(json.dumps(final_aggregated_result, ensure_ascii=False, indent=2))

    final_result: JsonDict = {
        "question": question,
        "task_mode": task_mode,
        "user_context": user_context or None,
        "has_user_context": bool(user_context),
        "retrieval_summary": {
            "retrieved_count": len(retrieved_evidence),
            "deduplicated_count": len(unique_retrieved_evidence),
            "total_evidence_count": len(all_evidence),
        },
        "total_evidence_count": len(enhanced_evidence),
        "agent_a_analysis": question_analysis,
        "agent_b_analysis": evidence_analysis,
        "agent_c_evaluation": evidence_evaluation,
        "agent_d_fusion": fusion_analysis,
        "final_report": final_report,
        "direct_llm_result": direct_llm_result,
        "final_aggregated_result": final_aggregated_result,
    }

    result_dir = Path(output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_file = result_dir / f"medar_qa_result_{timestamp}.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(final_result, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("✓ 单轮证据推理流程结束")
    print(f"✓ 输入证据数: {len(all_evidence)}")
    print(f"✓ 结果已保存到: {output_file}")
    print("=" * 80)

    return final_result
