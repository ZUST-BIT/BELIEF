import json
import os
import argparse
import time
from tqdm import tqdm
from langchain_openai import ChatOpenAI

# ========================
# 1. 配置与初始化
# ========================
try:
    from config import set_argument
except ImportError:
    print("❌ 找不到 config.py，请确保该文件在同一目录下")
    exit(1)

def init_llm():
    """初始化 GPT-4o-mini"""
    args = set_argument()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=args.api_key_gpt,
        base_url=args.api_url_gpt,
        # model_kwargs={"response_format": {"type": "json_object"}} # 可选：强制 JSON 模式
    )
    return llm

def extract_json_response(raw_text):
    """
    鲁棒的 JSON 提取函数
    """
    if not isinstance(raw_text, str):
        return {}
    
    cleaned_text = raw_text.strip()
    # 移除 Markdown 标记
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
    
    try:
        # 寻找最外层的 {}
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
    """加载数据集"""
    data = []
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []
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
# 2. 核心解题逻辑 (Zero-Shot)
# ========================
def solve_zero_shot(question_text, llm):
    """
    零样本回答：仅依赖模型内部知识/题目上下文
    """
    
    prompt = f"""
    You are a biomedical expert taking a closed-book exam.
    The task is to answer the question based ONLY on the information provided in the question text or your internal knowledge. 
    Do NOT use external tools or search.

    Question:
    {question_text}

    Options:
    A: yes
    B: no
    C: maybe

    Instructions:
    1. Read the question (and any accompanying context text) carefully.
    2. Select the single best answer (A, B, or C).
    3. Provide a brief explanation for your choice.
    4. You MUST output a valid JSON object strictly in the following format:
    {{
        "explanation": "Your brief explanation here.",
        "final_answer": "A" 
    }}
    (Note: final_answer must be "A", "B", or "C")
    """

    ans = "E" # Error
    exp = "Generation Error"
    raw_content = ""

    try:
        response = llm.invoke(prompt)
        raw_content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析 JSON
        parsed_json = extract_json_response(raw_content)
        
        if parsed_json:
            ans = parsed_json.get("final_answer", "E").upper().strip()
            exp = parsed_json.get("explanation", "")
            
            # 容错处理：如果模型输出了 YES/NO 而不是 A/B
            if len(ans) > 1:
                if "YES" in ans: ans = "A"
                elif "NO" in ans: ans = "B"
                elif "MAYBE" in ans: ans = "C"
        else:
            exp = f"JSON Parse Failed. Raw: {raw_content[:50]}..."

    except Exception as e:
        print(f"Error invoking model: {e}")
        exp = str(e)

    return ans, exp, raw_content

# ========================
# 3. 评测主流程
# ========================
def run_benchmark(input_path, output_path, num_samples):
    print(f"⏳ Initializing GPT-4o-mini Baseline...")
    llm = init_llm()

    # 准备目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载数据
    dataset = load_dataset(input_path)
    if not dataset: return

    # 断点续跑
    results = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f">>> Resuming: Found {len(results)} existing records.")
        except:
            results = []

    start_idx = len(results)
    total_to_run = min(num_samples, len(dataset))
    
    if start_idx >= total_to_run:
        print("✅ Target count reached.")
        return

    # 统计
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    answer_map = {"yes": "A", "no": "B", "maybe": "C"}

    print(f"🚀 Starting Zero-Shot Test: {start_idx + 1} to {total_to_run} ...")

    try:
        for i in tqdm(range(start_idx, total_to_run), ncols=100, desc="Testing"):
            item = dataset[i]
            
            # 字段兼容
            question = item.get("question") or item.get("QUESTION")
            
            # Ground Truth 处理
            gt_raw = str(item.get("final_decision") or item.get("answer") or "maybe").lower().strip()
            ground_truth = answer_map.get(gt_raw, "C")

            # ==== 核心：裸跑调用 ====
            t0 = time.time()
            pred, exp, raw_out = solve_zero_shot(question, llm)
            duration = time.time() - t0

            # 判分
            is_correct = (pred == ground_truth)
            if is_correct: correct_count += 1
            
            # 构造结果
            result_item = {
                "id": i,
                "model": "gpt-4o-mini-zero-shot",
                "question": question,
                "ground_truth": ground_truth,
                "prediction": pred,
                "is_correct": is_correct,
                "explanation": exp,
                "time_cost": round(duration, 2),
                "raw_llm_output": raw_out
            }
            results.append(result_item)

            # 打印简报
            icon = "✅" if is_correct else "❌"
            tqdm.write(f"[Q{i}] Pred: {pred} | GT: {ground_truth} | {icon}")

            # 定期保存 (每10条)
            if (i + 1) % 10 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    except KeyboardInterrupt:
        print("\n⚠️ User Interrupted.")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
    finally:
        # 最终保存
        current_total = len(results)
        final_acc = (correct_count / current_total) * 100 if current_total > 0 else 0
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n" + "="*40)
        print(f"🏁 Baseline Finished.")
        print(f"📊 Processed: {current_total} | Correct: {correct_count}")
        print(f"✅ Accuracy: {final_acc:.2f}%")
        print(f"📂 Saved to: {os.path.abspath(output_path)}")
        print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 输入数据集路径
    DEFAULT_INPUT = "D:/BitLabData/benchmark_data/PubMedQA/PubMedQA_test.json"  
    # 输出结果路径
    DEFAULT_OUTPUT = "results/pubmedqa_baseline2.0.json"
    
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    
    # 测试数量 (500)
    parser.add_argument("--num", type=int, default=500)
    
    args = parser.parse_args()

    if not args.input or not args.output:
        print("❌ 请先在代码中设置 DEFAULT_INPUT 和 DEFAULT_OUTPUT 路径，或者通过命令行参数传入。")
    else:
        run_benchmark(args.input, args.output, args.num)