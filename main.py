"""
MEDAR-QA 主流程 - 带多轮检索闭环
依次调用智能体A-E，实现动态证据推理系统
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from agents import AgentA, AgentB, AgentC, AgentD, AgentE, CompletenessController
from retriever import retrieve_process

def main():
    """主测试函数 - 带多轮检索闭环"""
    
    # question = """
    #     "question":"A 7-year-old boy is brought to his pediatrician’s office for a follow-up visit. He was diagnosed with asthma when he was 3 years old and has since been on treatment for the condition. He is currently on a β-agonist inhaler because of exacerbation of his symptoms. He has observed that his symptoms are more prominent in springtime, especially when the new flowers are blooming. His mother has a backyard garden and whenever he goes out to play there, he experiences chest tightness with associated shortness of breath. He has been advised to take more precaution during this seasonal change and to stay away from pollen. He is also being considered for an experimental therapy, which attenuates the activity of certain mediators which cause his asthmatic attack. The targeted mediator favors the class switching of antibodies. A reduction in this mechanism will eventually reduce the exaggerated response observed during his asthmatic attacks, even when exposed to an allergen. Which of the following mediators is described in this experimental study?"
    #     "options":{
    #     "A":"IL-2",
    #     "B":"IL-10",
    #     "C":"IL-13",
    #     "D":"IL-4"}
    # """
    question = """
            "question": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?",
            "options": {
            "A": "Inhibition of proteasome",
            "B": "Hyperstabilization of microtubules",
            "C": "Generation of free radicals",
            "D": "Cross-linking of DNA"
            },
    """
    context = """
            Chemotherapeutic agents used in the treatment of solid tumors often produce a variety of systemic adverse effects because they target rapidly dividing cells and may also affect certain normal tissues. In the management of bladder cancer, several classes of antineoplastic drugs are commonly used in neoadjuvant or adjuvant settings before surgical intervention. These include platinum-based compounds, taxanes, and other cytotoxic agents that interfere with cellular replication or survival pathways.
            One well-recognized complication of certain chemotherapy drugs is ototoxicity, which may manifest as tinnitus, hearing loss, or balance disturbances. The underlying mechanism is thought to involve damage to the sensory hair cells of the cochlea within the inner ear. In some cases, oxidative stress and mitochondrial injury contribute to this toxicity, particularly in patients receiving repeated cycles of chemotherapy.
            Different chemotherapeutic drugs exert their anticancer effects through distinct molecular mechanisms. For instance, some agents interfere with the proteasome, disrupting protein degradation pathways important for tumor cell survival. Others affect the microtubule network, preventing proper mitotic spindle formation and thereby inhibiting cell division. There are also drugs that induce cellular damage through the generation of reactive oxygen species, which can harm both cancer cells and certain normal tissues.
            Because chemotherapy regimens are selected based on tumor type and patient condition, clinicians must carefully weigh therapeutic benefits against the risk of adverse effects, including neurotoxicity and ototoxicity, when designing treatment strategies for cancer patients.
    """
    # question = "Colorectal cancer in young patients: is it a distinct clinical entity?"
    # context = """
    #         "The incidence of colorectal cancer in young patients is increasing. It remains unclear if the disease has unique features in this age group",
    #         "This was a single-center, retrospective cohort study which included patients diagnosed with colorectal cancer at age \u226440\u00a0years in 1997-2013 matched 1:2 by year of diagnosis with consecutive colorectal cancer patients diagnosed at age>50\u00a0years during the same period. Patients aged 41-50\u00a0years were not included in the study, to accentuate potential age-related differences. Clinicopathological characteristics, treatment, and outcome were compared between groups.",
    #         "The cohort included 330 patients, followed for a median time of 65.9\u00a0months (range 4.7-211). Several significant differences were noted. The younger group had a different ethnic composition. They had higher rates of family history of colorectal cancer (p\u00a0=\u00a00.003), hereditary colorectal cancer syndromes (p\u00a0<\u00a00.0001), and inflammatory bowel disease (p\u00a0=\u00a00.007), and a lower rate of polyps (p\u00a0<\u00a00.0001). They were more likely to present with stage III or IV disease (p\u00a0=\u00a00.001), angiolymphatic invasion, signet cell ring adenocarcinoma, and rectal tumors (p\u00a0=\u00a00.02). Younger patients more frequently received treatment. Young patients had a worse estimated 5-year disease-free survival rate (57.6\u00a0 vs. 70\u00a0%, p\u00a0=\u00a00.039), but this did not retain significance when analyzed by stage (p\u00a0=\u00a00.092). Estimated 5-year overall survival rates were 59.1 and 62.1\u00a0% in the younger and the control group, respectively (p\u00a0=\u00a00.565)."
    # """
    print("\n" + "="*80)
    print("MEDAR-QA 医学证据推理系统")
    print("="*80)
    print(f"\n问题: {question}\n")
    
    # 初始化智能体
    agent_a = AgentA()
    agent_b = AgentB()
    agent_c = AgentC()
    agent_d = AgentD()
    agent_e = AgentE()
    controller = CompletenessController()
    
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
        
        # 累积证据
        all_evidence.extend(retrieval_result)
        retrieval_history.append({
            "round": current_round,
            "evidence_count": len(retrieval_result),
            "total_evidence": len(all_evidence)
        })
        
        # 第3步：调用智能体B进行证据分析
        print(f"\n[轮次 {current_round} - 步骤 3/6] 智能体B：PICO提取与研究类型分类")
        print("-" * 80)
        agent_b_result = agent_b.run(question,all_evidence)
        print("\n智能体B分析结果:")
        print(json.dumps(agent_b_result, ensure_ascii=False, indent=2))
        
        # 第4步：调用智能体C进行证据评估
        print(f"\n[轮次 {current_round} - 步骤 4/6] 智能体C：证据可靠性评估与BPA计算")
        print("-" * 80)
        contextual_question = f"原问题{question}\n当前识别框架为{fod}。"
        question_pico_data = agent_a_result.get('pico_elements', {})
        # agent_c_result = agent_c.run(contextual_question, agent_b_result)
        agent_c_result = agent_c.run(
            hypothesis=contextual_question, 
            agent_b_result=agent_b_result,
            question_pico=question_pico_data,
            frame_of_discernment=fod,        # <--- 传入FoD，供规则引擎精确分配BPA
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
        bpa_summary = {
            "average_support": sum(b.get('support_hypothesis', 0) for b in bpa_list) / len(bpa_list) if bpa_list else 0,
            "average_against": sum(b.get('against_hypothesis', 0) for b in bpa_list) / len(bpa_list) if bpa_list else 0,
            "average_uncertainty": sum(b.get('uncertainty', 0) for b in bpa_list) / len(bpa_list) if bpa_list else 0,
            "bpa_count": len(bpa_list)
        }
        
        agent_e.add_reasoning_round(
            round_num=current_round,
            evidence_count=len(retrieval_result),
            bpa_summary=bpa_summary,
            note=f"第{current_round}轮推理完成"
        )
        
        # 完备性分析：判断是否需要继续检索
        print(f"\n[轮次 {current_round} - 完备性分析] 评估证据充分性")
        print("-" * 80)
        
        # 从agent_d_result中提取必要信息
        fused_bpa = agent_d_result.get('fusion_result', {}).get('fused_bpa', {
            'support_hypothesis': 0,
            'against_hypothesis': 0,
            'uncertainty': 1.0
        })
        belief_pl = agent_d_result.get('belief_plausibility', {
            'hypothesis_positive': {'belief': 0, 'plausibility': 1, 'uncertainty_interval': 1},
            'hypothesis_negative': {'belief': 0, 'plausibility': 1, 'uncertainty_interval': 1}
        })
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
                "score": ev_data.get('bpa_components', {}).get('support_hypothesis', 0),
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
        "final_report": agent_e_result
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
