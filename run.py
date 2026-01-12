import json
import os
from datetime import datetime
from agentic_tools import *
from context_manager import MultimodalContext
from retriever import retrieve_process
from query import query as initial_query

MAX_ROUNDS = 3
RELEVANCE_THRESHOLD = 6  # 相关性阈值

def run_qa_system(user_question, context='', save_log=True):
    """
    运行医学问答系统
    
    :param user_question: 用户的医学问题
    :param context: 自带的上下文（可选）
    :param save_log: 是否保存工作流日志
    :return: 最终答案和工作流报告
    """
    # 初始化智能体
    analyst = MuldimAnalyst()
    evaluator = EvidenceEvaluator()
    generator = GeneratorAgent()
    context_manager = MultimodalContext()

    round_idx = 1
    current_query = user_question
    analyst_instruction = ""  # 给分析模块的额外指令
    final_answer = None

    while round_idx <= MAX_ROUNDS:
        # print(f"\n=== 🌀 Round {round_idx} ===")
        
        # 0. 开始新一轮
        context_manager.start_round(round_idx, current_query)
        
        # 1. 检索
        retrieved_evidences = retrieve_process(current_query)
        
        # 1.1 第一轮时，如果用户提供了额外上下文，将其作为证据加入
        if round_idx == 1 and context:
            user_context_evidence = {
                "source_type": "User Provided Context",
                "content": context,
                "metadata": {"source": "user_input"}
            }
            retrieved_evidences.insert(0, user_context_evidence)  # 插入到最前面优先分析
        
        # print(f"📥 Retrieved {len(retrieved_evidences)} evidence blocks.")
        
        # 2. 逐条分析
        valid_count = 0
        for ev in retrieved_evidences:
            analyst_result = analyst.analyze_single(
                user_question,  # 始终使用原始问题进行分析
                ev,
                analyst_instruction
            )
            if analyst_result:
                score = analyst_result.get('relevance_score', 0)
                decision = analyst_result.get('decision', 'DISCARD')
                
                # 只保留高相关性且 KEEP 的证据
                if score >= RELEVANCE_THRESHOLD and decision == 'KEEP':
                    source_type = ev.get("source_type", "Unknown")
                    context_manager.add_analyst_result(analyst_result, source_type)
                    valid_count += 1
        
        # print(f"✅ Valid evidences after analysis: {valid_count}")
        
        # 3. 全局评估
        evaluation_data = context_manager.evidence_pool
        evaluation_result = evaluator.evaluate_global(
            user_question,  # 始终使用原始问题进行评估
            evaluation_data
        )
        
        # 4. 结束当前轮次，记录评估结果
        context_manager.end_round(evaluation_result)
        
        # 5. 决策处理
        status = evaluation_result.get("final_decision", "NO-GO")
        ds_analysis = evaluation_result.get("ds_analysis", {})
        
        # print(f"📊 Evaluation: {status}")
        # print(f"   - Belief Score: {ds_analysis.get('belief_score', 0):.2f}")
        # print(f"   - Uncertainty Gap: {ds_analysis.get('uncertainty_gap', 1.0):.2f}")
        # print(f"   - Conflict Detected: {ds_analysis.get('conflict_detected', False)}")
        
        if status == "GO":
            # 证据充足，生成最终答案
            print("\n🎯 Evidence sufficient, generating final answer...")
            evidence_text = context_manager.get_generator_input()
            print(f"\n🧾 Evidence provided to Generator:\n{evidence_text}\n")
            final_answer = generator.generate_answer(
                user_question,
                evidence_text
            )
            break
        else:
            # 证据不足，准备下一轮检索
            refinement = evaluation_result.get("refinement_strategy", {})
            next_queries = refinement.get("next_search_queries", [])
            feedback = refinement.get("feedback_to_analysis_agent", "")
            missing = refinement.get("missing_information", "")
            
            print(f"\n🔄 Need more evidence:")
            print(f"   - Missing: {missing}")
            print(f"   - Next Queries: {next_queries}")
            
            # 更新下一轮的查询和指令
            if next_queries:
                current_query = next_queries[0]  # 使用第一个推荐查询
            else:
                current_query = user_question  # 回退到原始问题
            
            if feedback:
                analyst_instruction = feedback
            
            round_idx += 1
    
    # 如果达到最大轮次仍未 GO，强制生成答案
    if final_answer is None:
        print("\n⚠️ Max rounds reached, forcing answer generation...")
        evidence_text = context_manager.get_generator_input()
        if evidence_text and evidence_text != "暂无相关证据。":
            final_answer = generator.generate_answer(
                user_question,
                evidence_text
            )
        else:
            final_answer = "抱歉，未能找到足够的证据来回答您的问题。请尝试重新描述问题或提供更多背景信息。"
    
    # 输出可解释性报告
    report = context_manager.get_explainability_report()
    print(f"\n{report}")
    
    # 保存工作流日志
    if save_log:
        _save_workflow_log(context_manager, user_question, final_answer)
    

    return final_answer
    # return {
    #     "answer": final_answer,
    #     "evidence_pool": context_manager.evidence_pool,
    #     "workflow_log": context_manager.export_workflow_log()
    # }


def _save_workflow_log(context_manager, question, answer):
    """
    保存工作流日志到文件
    
    :param context_manager: 上下文管理器
    :param question: 用户问题
    :param answer: 最终答案
    """
    log_dir = "workflow_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存 JSON 格式
    json_log = {
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        **context_manager.export_workflow_log()
    }
    json_path = os.path.join(log_dir, f"workflow_log_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_log, f, ensure_ascii=False, indent=2)
    
    # 保存可读性报告
    txt_path = os.path.join(log_dir, f"workflow_log_{timestamp}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Question: {question}\n")
        f.write(f"{'='*50}\n")
        f.write(context_manager.get_explainability_report())
        f.write(f"\n{'='*50}\n")
        f.write(f"Final Answer:\n{answer}\n")
    
    # print(f"\n📁 Workflow log saved: {json_path}")


# --- 主程序入口 ---
if __name__ == '__main__':
    question = initial_query
    context = ""
    result = run_qa_system(question, context)
    
    print("\n" + "="*60)
    print("🎯 Final Answer:")
    print("="*60)
    print(type(result))
    print(result)
