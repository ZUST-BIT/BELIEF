import json
import os
import argparse
import time
from tqdm import tqdm

# ========================
# 1. 模块导入与环境检查
# ========================
try:
    from retriever import retrieve_process
    from agentic_tools import GeneratorAgent, ReasonerAgent, ReflectorAgent
    from config import set_argument
except ImportError as e:
    print(f"❌ Module import failed: {e}")
    print("Please make sure you are in the correct directory (where 'main.py' exists).")
    exit(1)

# ========================
# 2. 辅助函数
# ========================
def init_agents():
    """初始化智能体"""
    _ = set_argument() # 确保配置加载
    generator = GeneratorAgent()
    reasoner = ReasonerAgent()
    reflector = ReflectorAgent()
    return generator, reasoner, reflector

def extract_json_response(raw_text):
    """
    从 LLM 的字符串输出中提取并解析 JSON
    处理 Markdown 代码块 (```json ... ```) 的情况
    """
    if not isinstance(raw_text, str):
        return {}
        
    cleaned_text = raw_text.strip()
    
    # 移除 Markdown 标记
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
    
    try:
        # 尝试解析 JSON
        start_idx = cleaned_text.find("{")
        end_idx = cleaned_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = cleaned_text[start_idx : end_idx + 1]
            return json.loads(json_str)
        else:
            return {}
    except json.JSONDecodeError:
        return {}

def load_dataset(file_path):
    """加载数据集 (支持 JSON List 和 JSONL)"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []

    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
            else:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        print(f">>> Loaded {len(data)} samples from {os.path.basename(file_path)}")
        return data
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return []

# ========================
# 3. 核心解题逻辑
# ========================
def solve_pubmedqa(question_text, generator, reasoner, reflector):
    """
    执行 RAG 流程: [检索-反思循环] -> 推理 -> 生成 -> 解析 JSON
    """
    
    # --- A. 检索-反思循环 (Retrieval with Reflection) ---
    # 【新增】这里不再只检索一次，而是进入反思循环
    
    current_search_query = question_text
    full_evidence_text = ""  # 用于累积所有轮次的证据
    retrieved_queries = set() # 防止重复检索
    retrieved_queries.add(current_search_query)
    reflector_logs = []      # 记录反思过程用于 Context
    
    MAX_RETRIES = 2  # 最大重试次数 (0, 1, 2 共3次机会)
    retry_count = 0
    
    while retry_count <= MAX_RETRIES:
        # 1. 执行检索
        try:
            # 检索新证据
            new_evidence = retrieve_process(current_search_query)
            # 将新证据追加到总证据中 (标注来源查询)
            full_evidence_text += f"\n\n[Evidence from query: '{current_search_query}']\n{new_evidence}"
        except Exception as e:
            full_evidence_text += f"\nRetrieval Error for {current_search_query}: {str(e)}"
        
        # 2. 调用 Reflector 评估
        # Reflector 会阅读当前积累的所有证据，判断是否足够回答 user_query
        try:
            evaluation = reflector.evaluate(question_text, full_evidence_text)
            status = evaluation.get("status", "pass")
            next_query = evaluation.get("new_query", "")
            feedback = evaluation.get("feedback", "")
            
            log_entry = f"Iter {retry_count}: Status={status}, NextQuery='{next_query}'"
            reflector_logs.append(log_entry)
            
            if status == "pass":
                # 证据充足，跳出循环
                break
            else:
                # 证据不足，准备下一次检索
                if retry_count < MAX_RETRIES and next_query:
                    if next_query in retrieved_queries:
                        # 如果生成了重复的查询，停止以避免死循环
                        reflector_logs.append(f"Stopped: Duplicate query '{next_query}'")
                        break
                    
                    # 更新查询词，进入下一轮
                    current_search_query = next_query
                    retrieved_queries.add(next_query)
                    retry_count += 1
                else:
                    # 次数用尽或无法生成新问题
                    break
                    
        except Exception as e:
            print(f"Reflector Error: {e}")
            break

    # --- B. 推理 (Reasoning) ---
    try:
        # 【修改】使用累积的 full_evidence_text 进行推理
        reasoning_content = reasoner.analyze(question_text, full_evidence_text)
    except Exception as e:
        reasoning_content = f"Reasoning failed: {str(e)}"

    # 构造完整上下文用于记录和 Prompt
    full_context = (
        f"=== Reflector Logs ===\n" + "\n".join(reflector_logs) + "\n\n"
        f"=== Accumulated Evidence ===\n{full_evidence_text}\n\n"
        f"=== Reasoning Insights ===\n{reasoning_content}"
    )

    # --- C. 生成 (Generation) ---
    # 截断证据防止 Token 溢出 (保留最近 10000 字符)
    prompt = f"""
    You are a biomedical expert answering PubMedQA.
    The task is to answer the question with "yes", "no", or "maybe".

    Question:
    {question_text}

    Options:
    A: yes
    B: no
    C: maybe

    Retrieved Evidence & Reasoning:
    {full_context[:12000]} 

    Instructions:
    1. Analyze the evidence carefully.
    2. Select the single best answer (A, B, or C).
    3. Provide a brief explanation.
    4. You MUST output a valid JSON object strictly in the following format:
    {{
        "explanation": "Your brief explanation here.",
        "final_answer": "A" 
    }}
    (Note: final_answer must be one of "A", "B", or "C")
    """

    # --- D. 调用模型与解析 ---
    ans = "E" # Default Error
    exp = "Error in generation"
    raw_content = ""

    try:
        response = generator.llm.invoke(prompt)
        raw_content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析 JSON
        parsed_json = extract_json_response(raw_content)
        
        if parsed_json:
            ans = parsed_json.get("final_answer", "E").upper().strip()
            exp = parsed_json.get("explanation", "")
            
            # 容错处理
            if len(ans) > 1:
                if "YES" in ans: ans = "A"
                elif "NO" in ans: ans = "B"
                elif "MAYBE" in ans: ans = "C"
        else:
            exp = f"JSON Parse Failed. Raw: {raw_content[:100]}"

    except Exception as e:
        print(f"Error invoking model: {e}")

    return ans, exp, full_context, raw_content

def run_benchmark(input_path, output_path, num_samples):
    # 初始化 (增加 reflector)
    generator, reasoner, reflector = init_agents()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载数据
    dataset = load_dataset(input_path)
    if not dataset: return

    # 断点续跑逻辑
    results = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f">>> Resuming from {len(results)} existing results.")
        except Exception:
            results = []

    start_idx = len(results)
    total_to_run = min(num_samples, len(dataset))
    
    if start_idx >= total_to_run:
        print("✅ All requested samples have already been processed.")
        return

    # 统计正确数
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    answer_map = {"yes": "A", "no": "B", "maybe": "C"}

    print(f"🚀 Starting Test: Processing {start_idx + 1} to {total_to_run} ...")

    try:
        for i in tqdm(range(start_idx, total_to_run), ncols=100, desc="Testing"):
            item = dataset[i]
            
            # 字段兼容处理
            question = item.get("question") or item.get("QUESTION")
            
            # Ground Truth 处理
            gt_raw = str(item.get("final_decision") or item.get("answer") or "maybe").lower().strip()
            ground_truth = answer_map.get(gt_raw, "C")

            # ==== 核心调用 (传入 reflector) ====
            t0 = time.time()
            pred, exp, ctx, raw_out = solve_pubmedqa(question, generator, reasoner, reflector)
            duration = time.time() - t0

            # 判分
            is_correct = (pred == ground_truth)
            if is_correct: correct_count += 1
            
            # 构造结果
            result_item = {
                "id": i,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": pred,
                "is_correct": is_correct,
                "explanation": exp,       
                "time_cost": round(duration, 2),
                "full_context": ctx,      
                "raw_llm_output": raw_out 
            }
            results.append(result_item)

            # 实时日志 (Pred / GT)
            icon = "✅" if is_correct else "❌"
            tqdm.write(f"[Q{i}] Pred: {pred} | GT: {ground_truth} | {icon}")

            # 定期保存 (每5条)
            if (i + 1) % 5 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    except KeyboardInterrupt:
        print("\n⚠️ User interrupted. Saving progress...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        # 最终保存
        current_total = len(results)
        final_acc = (correct_count / current_total) * 100 if current_total > 0 else 0
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n" + "="*40)
        print(f"🏁 Test Finished")
        print(f"📊 Processed: {current_total} | Correct: {correct_count}")
        print(f"✅ Accuracy: {final_acc:.2f}%")
        print(f"📂 Saved to: {os.path.abspath(output_path)}")
        print("="*40)

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数，方便你在服务器或不同配置下运行
    parser = argparse.ArgumentParser(description="Run PubMedQA Benchmark")
    
    # 默认路径配置 (你可以在这里修改默认值)
    DEFAULT_INPUT = "D:/BitLabData/benchmark_data/PubMedQA/PubMedQA_test.json"
    DEFAULT_OUTPUT = "results/pubmedqa_medar(4omini2.0).json"
    
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to input dataset")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to output result JSON")
    parser.add_argument("--num", type=int, default=500, help="Max number of samples to test")
    
    args = parser.parse_args()

    run_benchmark(
        input_path=args.input,
        output_path=args.output,
        num_samples=args.num
    )