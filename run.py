import os
import time
import shutil
from query import query as original_user_query
from format_config import FormatType
from agentic_tools import format_selector,generate_code, CoderAgent, CodeExecutor, GeneratorAgent, ReflectorAgent, MultimodalContext, ReasonerAgent
from retriever import retrieve_process
from intent_router import IntentRouter
if __name__ == '__main__':
    evidence_final = ""
    retry_count = 0
    
    context = MultimodalContext()
    reflector = ReflectorAgent()
    reasoner = ReasonerAgent()
    executor = CodeExecutor()
    generator = GeneratorAgent()
    intent_router = IntentRouter()
    
    intent_res = intent_router.intent_router(question=original_user_query, context=original_user_query)
    need_retrieval = intent_res.get("need_retrieval", "YES").strip().upper()
    rewritten_query = intent_res.get("rewritten_query", original_user_query)
    analysis = intent_res.get("analysis", "")
    current_search_query = rewritten_query if need_retrieval == "YES" else original_user_query
    context.add_query(current_search_query)
    if need_retrieval == "YES":
        print(f"\n🔍 >>> Phase 1: Entering Retrieval Loop (Query: {current_search_query})...")
        MAX_RETRIES = 2 # 最大重试次数
        while retry_count <= MAX_RETRIES:
            print(f"\n🔄 --- Iteration {retry_count + 1} ---")

            new_evidence = retrieve_process(current_search_query)
            context.add_evidence(new_evidence)
            full_evidence = context.get_consolidated_text()

            evaluation = reflector.evaluate(original_user_query, full_evidence)
            status = evaluation.get("status", "pass")
            feedback = evaluation.get("feedback", "")
            next_query = evaluation.get("new_query", "")

            if status == "pass":
                print(">>> Evidence is sufficient to answer the question.")
                break
            else:
                if retry_count < MAX_RETRIES and next_query:
                    if next_query in context.query_history:
                        print(">>> New query has already been searched. Stopping to avoid loops.")
                        break
                    print(f">>> Preparing next hop with query: '{next_query}'")
                    current_search_query = next_query
                    context.add_query(current_search_query)
                    retry_count += 1
                else:
                    print("⚠️ Max retries reached or no new query generated. Proceeding with what we have.")
                    break
    else:
        print("\n🚫 >>> Phase 1: Retrieval Skipped (Context Self-Contained).")
        context.add_evidence("No external retrieval performed (Intent Router decision).")
    
    # Phase 2: 推理 (Reasoner)
    raw_retrieved_evidence = context.get_consolidated_text()
    print(">>> Reasoner is analyzing connections in the evidence...")
    try:
        insights = reasoner.analyze(original_user_query, original_user_query, raw_retrieved_evidence)
        context.add_reasoning(insights)
    except Exception as e:
        print(f">>> Reasoner failed to analyze the evidence: {e}")
    
    # 获取包含推理内容的完整证据
    full_evidence_with_insights = context.get_consolidated_text()

    # Phase 3: 可视化 (Visualization)
    decision = format_selector(original_user_query, full_evidence_with_insights)
    selected_fmt = decision.get("selected_format", FormatType.TEXT.value)
    reasoning = decision.get("reasoning", "")
    
    print(f">>> Decison: [{selected_fmt.upper()}] because {reasoning}")
    
    final_image_path = None
    
    if selected_fmt in [FormatType.CHART.value, FormatType.GRAPH.value]:
        print(f">>> Format '{selected_fmt}' requires visualization . Invoke coder...")
        try:
            generated_code = generate_code(selected_fmt, full_evidence_with_insights)
            temp_output_file = executor.execute(generated_code)
            
            if temp_output_file and os.path.exists(temp_output_file):
                save_dir = os.path.join(os.getcwd(), "visual_outputs")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                new_filename = f"viz_{timestamp}.png"
                target_path = os.path.join(save_dir, new_filename)
                try:
                    shutil.move(temp_output_file, target_path)
                    print(f">>> Visualization saved to {target_path}")
                    final_image_path = target_path
                    context.add_image(target_path)
                except Exception as e:
                    print(f">>> Failed to save visualization to {target_path}: {e}")
            else:
                print(">>> Failed to generate visualization")
        except Exception as e:
            print(f">>> Failed to generate visualization: {e}")
    elif selected_fmt == FormatType.TEXT.value:
        print(">>> Selected format is TEXT. No visualization needed.")
    else:
        print(f">>> Unsupported format: {selected_fmt}")
    
    print(">>> Process completed.")

    # Phase 4: 生成最终回答 (Generator)
    print(">>> Generating final answer...")
    final_answer = generator.generate_answer(
        query=original_user_query,
        evidence_text=full_evidence_with_insights[:10000], # 传入包含 Reasoning 的证据
        image_path=final_image_path
    )
    print("\n" + "=" * 50)
    print("🤖 MEDAR-QA Final Answer:")
    print("=" * 50)
    print(final_answer)
    print("=" * 50)