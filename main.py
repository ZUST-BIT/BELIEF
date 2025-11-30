import os,time,shutil
from intent_router import IntentRouter
from neo4j import Neo4jManager
from omic import OmicData
from pubmed import hcc_pubmed_data, bio_pubmed_data
from query import query as original_user_query
from util.data_refiner import EvidenceRefiner
from format_config import FormatType
from visual_module import FormatSelector, CoderAgent, CodeExecutor, GeneratorAgent, ReflectorAgent, MultimodalContext, ReasonerAgent

def retrieve_process(current_search_query):
    print(f">>> Retrieving data for query: {current_search_query}")
    intent_router = IntentRouter()
    neo4j_manager = Neo4jManager()
    omic_data_source = OmicData()
    hcc_pubmed = hcc_pubmed_data()
    bio_pubmed = bio_pubmed_data()
    refiner = EvidenceRefiner()

    query_intent = intent_router.intent_router(current_search_query)
    entity_list = query_intent['extracted_entities']

    # retrieve data
    neo4j_data = neo4j_manager.query_node_info(entity_list)
    omic_resutls = omic_data_source.search_omic_data(current_search_query)
    hcc_data = hcc_pubmed.search_hcc_faiss(current_search_query)
    bio_data = bio_pubmed.search_bio_faiss(current_search_query)

    # refine data
    refined_data = refiner.run(
        kg_data=neo4j_data, 
        omic_data=omic_resutls, 
        hcc_data=hcc_data, 
        bio_data=bio_data
    )

    return refined_data

if __name__ == '__main__':
    evidence_final = ""
    retry_count = 0
    # init
    context = MultimodalContext()
    reflector = ReflectorAgent()
    reasoner = ReasonerAgent()
    selector = FormatSelector()
    coder = CoderAgent()
    executor = CodeExecutor()
    generator = GeneratorAgent()
    current_search_query = original_user_query
    context.add_query(current_search_query)
    MAX_RETRIES = 2 # 最大重试次数
    while retry_count <= MAX_RETRIES:
        print(f"\n🔄 --- Iteration {retry_count + 1} ---")

        new_evidence = retrieve_process(current_search_query)
        print(f"Fetched Length: {len(new_evidence)}")

        context.add_evidence(new_evidence)

        full_evidence = context.get_consolidated_text()

        print(">>> Reflector: Evaluating the sufficiency of the evidence...")
        evaluation = reflector.evaluate(original_user_query, full_evidence)

        status = evaluation.get("status", "pass")
        feedback = evaluation.get("feedback", "")
        next_query = evaluation.get("new_query", "")

        print(f">>> Evaluation: [{status.upper()}]")
        print(f">>> Feedback: {feedback}")

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
    raw_evidence = context.get_consolidated_text()
    print(">>> Reasoner is analyzing connections in the evidence...")
    try:
        insights = reasoner.analyze(original_user_query, raw_evidence)
        print(f"\n[Reasoner Output]:\n{insights}\n")
        context.add_reasoning(insights)
    except Exception as e:
        print(f">>> Reasoner failed to analyze the evidence: {e}")
    print("--- Debug: Reviewing Context ---")
    context.show_status()
    full_evidence = context.get_consolidated_text()
    # select
    decision = selector.select(original_user_query, full_evidence)
    selected_fmt = decision.get("selected_format", FormatType.TEXT.value)
    reasoning = decision.get("reasoning", "")
    print(f">>> Decison: [{selected_fmt.upper()}] because {reasoning}")
    final_image_path = None
    if selected_fmt in [FormatType.CHART.value, FormatType.GRAPH.value]:
        print(f">>> Format '{selected_fmt}' requires visualization . Invoke coder...")
        try:
            generated_code = coder.generate_code(selected_fmt, full_evidence)
            # execute
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
                    final_image_path = target_path  # 路径传值
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

    # generate final answer
    print(">>> Generating final answer...")
    final_answer = generator.generate_answer(
        query=original_user_query,
        evidence_text=full_evidence[:10000],
        image_path=final_image_path
    )
    print("\n" + "=" * 50)
    print("🤖 MEDAR-QA Final Answer:")
    print("=" * 50)
    print(final_answer)
    print("=" * 50)

