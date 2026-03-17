"""
MEDAR-QA 主流程 - 带多轮检索闭环
依次调用智能体A-E，实现动态证据推理系统
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from agents import AgentA, AgentB, AgentC, AgentD, AgentE, CompletenessController, AgentDirectLLM, AgentFinalAggregator
from retriever import retrieve_process


def _dedup_against_user_context(papers: list, user_context_text: str, fingerprint_len: int = 60) -> list:
    """
    过滤掉与用户上下文高度重叠的检索论文，防止同一研究被双重计数。
    检测方式：取论文 Summary 正文首 fingerprint_len 字符，判断是否出现在 user_context 中。
    """
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
                # 移除 "BACKGROUND:" "RESULTS:" 等段落标签
                if ':' in body[:20]:
                    body = body[body.index(':') + 1:].strip()
                fingerprint = body[:fingerprint_len].lower()
                if fingerprint and fingerprint in ctx_lower:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(paper)
    return unique


def main():
    """主测试函数 - 带多轮检索闭环"""
    
    # question = """
    #     "question":"A 3-week-old male newborn is brought to the physician because of an inward turning of his left forefoot. He was born at 38 weeks' gestation by cesarean section because of breech presentation. The pregnancy was complicated by oligohydramnios. Examination shows concavity of the medial border of the left foot with a skin crease just below the ball of the great toe. The lateral border of the left foot is convex. The heel is in neutral position. Tickling the lateral border of the foot leads to correction of the deformity. The remainder of the examination shows no abnormalities. X-ray of the left foot shows an increased angle between the 1st and 2nd metatarsal bones. Which of the following is the most appropriate next step in the management of this patient?"
    #     "options":{
    #     "A":"Foot abduction brace",
    #     "B":"Arthrodesis of the forefoot",
    #     "C":"Reassurance",
    #     "D":"Tarsometatarsal capsulotomy"
    # }
    # """
    # question = """
    #         "question": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?",
    #         "options": {
    #         "A": "Inhibition of proteasome",
    #         "B": "Hyperstabilization of microtubules",
    #         "C": "Generation of free radicals",
    #         "D": "Cross-linking of DNA"
    #         },
    # """
    context = ""
    question = "Continuation of pregnancy after antenatal corticosteroid administration: opportunity for rescue?"
    context = """
            "To determine the duration of continuing pregnancy after antenatal corticosteroid (AC) administration and to evaluate the potential opportunity for rescue AC.",
            "Retrospective analysis of women at 24-32 weeks' gestation who received AC at one institution.",
            "Six hundred ninety-two women received AC. Two hundred forty-seven (35.7%) delivered at>or = 34 weeks' gestation. Three hundred twenty-one (46.4%) delivered within 1 week of AC; 92 of those women (13.3%) delivered within 24 hours. Only 124 (17.9%) remained pregnant 1 week after AC and delivered at<34 weeks. The latter were compared to women delivering>2 week after AC but>or = 34 weeks. More likely to deliver at<34 weeks were those women who received AC for premature preterm rupture of membranes (OR 3.83, 95% CI 2.06-7.17), twins (OR 2.90, 95% CI 1.42-5.95) or before 28 weeks (OR 2.21, 95% CI 1.38-3.52)."
    """
    # ======================================================
    # 直接LLM分支开关
    # 设置为 True  → 在D-S流程之外并行运行直接LLM推理分支，
    #                     并在最后由聚合智能体合并两条分支的结果。
    # 设置为 False → 仅运行现有D-S流程，跳过聚合步骤。
    ENABLE_DIRECT_LLM_BRANCH: bool = True
    # ======================================================

    print("\n" + "="*80)
    print("MEDAR-QA 医学证据推理系统")
    print("="*80)
    print(f"\n问题: {question}\n")
    if ENABLE_DIRECT_LLM_BRANCH:
        print("✔ 直接LLM分支开关：已开启（将并行D-S分支与直接LLM分支，最后聚合）")
    else:
        print("✓ 直接LLM分支开关：已关闭（仅运行现有D-S流程）")
    
    # 初始化智能体
    agent_a = AgentA()
    agent_b = AgentB()
    agent_c = AgentC()
    agent_d = AgentD()
    agent_e = AgentE()
    controller = CompletenessController()
    # 如果开关开启，初始化新增的两个智能体
    agent_direct_llm = AgentDirectLLM() if ENABLE_DIRECT_LLM_BRANCH else None
    agent_final_agg = AgentFinalAggregator() if ENABLE_DIRECT_LLM_BRANCH else None
    direct_llm_result = None   # 直接LLM分支的输出，默认为None
    
    # 第1步：调用智能体A进行问题分析
    print("\n[步骤 1/6] 智能体A：问题分析与实体提取")
    print("-" * 80)
    agent_a_result = agent_a.run(question)
    print("\n智能体A分析结果:")
    print(json.dumps(agent_a_result, ensure_ascii=False, indent=2))
    fod = agent_a_result.get('frame_of_discernment', ['H', '¬H'])
    # 多轮检索闭环
    max_rounds = 1
    current_round = 1
    all_evidence = []
    retrieval_history = []
    
    # 处理用户自带的上下文：如果有则作为初始证据片段
    if context and context.strip():
        user_context_evidence = {
            "source": "user_provided_context",
            "content": context.strip(),
            "score": 1.0,  # 用户提供的上下文给予最高相关性分数
            "type": "user_context",
            "metadata": {
                "is_user_provided": True,
                "description": "用户提供的背景上下文信息"
            }
        }
        all_evidence.append(user_context_evidence)
    
    while current_round <= max_rounds:
        print(f"\n{'='*80}")
        print(f"第 {current_round} 轮证据推理")
        print(f"{'='*80}")
        
        # 第2步：基于智能体A的结果进行知识检索
        print(f"\n[轮次 {current_round} - 步骤 2/6] 知识检索：结合实体进行证据检索")
        print("-" * 80)
        retrieval_result = retrieve_process(question, agent_a_result)
        
        # 累积证据（过滤与用户上下文重叠的论文，避免双重计数）
        user_ctx_text = context if context and context.strip() else ""
        deduped = _dedup_against_user_context(retrieval_result, user_ctx_text)
        skipped = len(retrieval_result) - len(deduped)
        if skipped:
            print(f"[去重] 检测到 {skipped} 条检索结果与用户上下文高度重叠，已过滤")
        all_evidence.extend(deduped)
        retrieval_history.append({
            "round": current_round,
            "evidence_count": len(retrieval_result),
            "total_evidence": len(all_evidence)
        })
        
        # 第3步：调用智能体B进行证据分析
        print(f"\n[轮次 {current_round} - 步骤 3/6] 智能体B：PICO提取与研究类型分类")
        print("-" * 80)
        agent_b_result = agent_b.run(question, all_evidence)
        print("\n智能体B分析结果:")
        print(json.dumps(agent_b_result, ensure_ascii=False, indent=2))

        # ---- 直接LLM分支（可选）----
        if ENABLE_DIRECT_LLM_BRANCH and current_round == 1:
            # 只在第1轮运行一次；如需每轮都运行可将条件去掉
            print(f"\n[轮次 {current_round} - 直接LLM分支] 开始直接LLM推理")
            print("-" * 80)
            direct_llm_result = agent_direct_llm.run(
                question=question,
                agent_b_result=agent_b_result,
                task_mode=agent_a_result.get('task_mode', 'SELECTION'),
                verbose=True
            )
            print("\n直接LLM分支结果:")
            print(json.dumps(direct_llm_result, ensure_ascii=False, indent=2))
        
        # 第4步：调用智能体C进行证据评估
        print(f"\n[轮次 {current_round} - 步骤 4/6] 智能体C：证据可靠性评估与BPA计算")
        print("-" * 80)
        contextual_question = f"原问题{question}\n当前识别框架为{fod}。"
        # 新版 Prompt_A 将 PICO 嵌套在 extraction.elements 中
        question_pico_data = agent_a_result.get('extraction', {}).get('elements', {})
        # agent_c_result = agent_c.run(contextual_question, agent_b_result)
        agent_c_result = agent_c.run(
            hypothesis=contextual_question, 
            agent_b_result=agent_b_result,
            question_pico=question_pico_data,
            frame_of_discernment=fod,     
            verbose=False
        )
        print("\n智能体C评估结果:")
        print(json.dumps(agent_c_result, ensure_ascii=False, indent=2))
        # print(f"生成 {len(agent_c_result.get('bpa_list', []))} 个BPA")
        
        # 第5步：调用智能体D进行证据融合
        print(f"\n[轮次 {current_round} - 步骤 5/6] 智能体D：多证据融合与决策")
        print("-" * 80)
        
        # 提取FoD和BPA列表
        # fod = agent_a_result.get('frame_of_discernment', ['H', '¬H'])
        bpa_list = agent_c_result.get('bpa_list', [])
        # 修改后：传入包含内容信息的 evaluations
        evaluations_for_d = agent_c_result.get('evaluations', [])
        # agent_d_result = agent_d.run(question, fod, evaluations_for_d)
        if not bpa_list:
            print("⚠️ 没有有效的BPA，跳过融合步骤")
            agent_d_result = {
                "note": "No valid BPA for fusion",
                "final_decision": {
                    "decision": "INSUFFICIENT_EVIDENCE",
                    "confidence": 0.0,
                    "reason": "没有足够的有效证据进行推理"
                }
            }
        else:
            agent_d_result = agent_d.run(question, fod, bpa_list)
            print("\n智能体D融合结果:")
            print(json.dumps(agent_d_result, ensure_ascii=False, indent=2))
        
        # 记录本轮推理结果到Agent E
        # bpa_list 现在是 FoD 键格式，汇总每个选项的平均质量
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
        
        # 完备性分析：判断是否需要继续检索
        print(f"\n[轮次 {current_round} - 完备性分析] 评估证据充分性")
        print("-" * 80)
        
        # 从agent_d_result中提取必要信息（新格式：FoD键BPA）
        raw_fused = agent_d_result.get('fusion_result', {}).get('fused_bpa', {})
        opt_masses = {k: v for k, v in raw_fused.items() if k != 'uncertainty_theta'}
        top_mass   = max(opt_masses.values()) if opt_masses else 0.0
        u_mass     = raw_fused.get('uncertainty_theta', 1.0)
        # 为 CompletenessController 构造兼容的二元视图
        fused_bpa = {
            'support_hypothesis': top_mass,
            'against_hypothesis': round(sum(opt_masses.values()) - top_mass, 4),
            'uncertainty': u_mass
        }
        belief_pl = {
            'hypothesis_positive': {
                'belief': top_mass,
                'plausibility': round(top_mass + u_mass, 4),
                'uncertainty_interval': u_mass
            },
            'hypothesis_negative': {
                'belief': round(max(0.0, 1.0 - top_mass - u_mass), 4),
                'plausibility': round(1.0 - top_mass, 4),
                'uncertainty_interval': u_mass
            }
        }
        conflict_coef = agent_d_result.get('fusion_result', {}).get('conflict_coefficient', 0)
        
        completeness_result = controller.analyze_completeness(fused_bpa, belief_pl, conflict_coef)
        
        print(f"当前状态: {completeness_result['state']}")
        print(f"Deng熵: {completeness_result.get('entropy', 0):.4f}")
        print(f"决策建议: {completeness_result['action']}")
        print(f"是否继续检索: {completeness_result['should_continue']}")
        print(f"理由: {completeness_result['reason']}")
        if 'suggestion' in completeness_result:
            print(f"建议: {completeness_result['suggestion']}")
        
        # 判断是否继续下一轮
        if not completeness_result['should_continue']:
            print(f"\n✓ 证据充分性满足要求，终止检索")
            break
        
        if current_round >= max_rounds:
            print(f"\n✓ 已达到最大轮次限制 ({max_rounds})，终止检索")
            break
        
        print(f"\n→ 证据不充分，需要继续检索...")
        current_round += 1
    
    # 第6步：调用智能体E生成最终报告
    print(f"\n{'='*80}")
    print("[步骤 6/6] 智能体E：生成最终医学证据报告")
    print("="*80)

    agent_d_logic_note = f"""
    [Agent D Logic Trace]:
    - System Decision: {agent_d_result.get('final_decision', {}).get('decision')}
    - Evidence Mapping: Agent D identified specific evidence IDs as supporting the decision. 
      (Agent E MUST verify if this mapping aligns with clinical symptoms).
    """
    agent_e.add_reasoning_round(99, 0, {}, note=agent_d_logic_note)

    # 2. 拼接增强型证据列表 (Enhanced Evidence List)
    enhanced_evidence_input = []
    c_evaluations = agent_c_result.get('evaluations', [])
    if c_evaluations:
        print(f"正在组装 {len(c_evaluations)} 条增强型证据报告...")
        for ev in c_evaluations:
            ev_data = ev.get('evaluation', {})
            # 优先使用 Agent C 生成的富文本报告
            rich_content = ev_data.get('content_for_generator')
            
            # 兜底：如果没有富文本，使用原始摘要
            if not rich_content:
                rich_content = f"[Raw Snippet]: {ev_data.get('processed_input_snippet', 'N/A')}"
            
            enhanced_evidence_input.append({
                "source": ev.get('source_type', 'Unknown'),
                "content": rich_content,  # <--- 替换为富文本
                # 取 bpa_distribution 中所有选项质量的最大值作为得分
                "score": max(
                    (v for k, v in ev_data.get('bpa_distribution', {}).items()
                     if k != 'uncertainty_theta'),
                    default=0.0
                ),
                "type": "analyzed_report",
                "metadata": ev.get('metadata', {})
            })
    else:
        print("⚠️ 警告：未检测到增强分析，回退到原始证据。")
        enhanced_evidence_input = all_evidence
    final_decision = agent_d_result.get('final_decision', {
        "decision": "UNCERTAIN",
        "confidence": 0.0,
        "reason": "未获得决策结果"
    })
    
    fusion_result = agent_d_result.get('fusion_result', {})
    belief_analysis = agent_d_result.get('belief_plausibility', {})

    agent_e_result = agent_e.run(
        question=question,
        final_decision=final_decision,
        fusion_result=fusion_result,
        # belief_analysis=belief_analysis,
        evidence_list=enhanced_evidence_input
    )
    print("\n最终报告:")
    print(json.dumps(agent_e_result, ensure_ascii=False, indent=2))

    # ---- 最终聚合（可选）----
    final_aggregated_result = None
    if ENABLE_DIRECT_LLM_BRANCH and direct_llm_result is not None:
        print(f"\n{'='*80}")
        print("[最终聚合] 综合DS分支与直接LLM分支结果")
        print("="*80)
        final_aggregated_result = agent_final_agg.run(
            question=question,
            ds_result=agent_e_result,
            direct_llm_result=direct_llm_result,
            task_mode=agent_a_result.get('task_mode', 'SELECTION'),
            verbose=True
        )
        print("\n最终聚合结果:")
        print(json.dumps(final_aggregated_result, ensure_ascii=False, indent=2))

    # 汇总所有结果
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
        # "completeness_analysis": completeness_result,
        "final_report": agent_e_result,
        # 直接LLM分支相关字段（开关关闭时均为None）
        "direct_llm_result": direct_llm_result,
        "final_aggregated_result": final_aggregated_result,
    }
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"TEST_RESULTS/test_demo_result_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print(f"✓ 完整推理流程结束")
    print(f"✓ 总轮次: {current_round}")
    print(f"✓ 总证据数: {len(all_evidence)}")
    print(f"✓ 结果已保存到: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
