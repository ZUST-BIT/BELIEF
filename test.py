import json
import os
from tqdm import tqdm
from datasets import load_dataset
from langchain_openai import ChatOpenAI

# 1. 导入配置 (复用你的 config)
from config import set_argument
args = set_argument()

# ==========================================
# 配置你的基准模型 (Baseline Model)
# ==========================================
# 你可以在这里切换模型，比如 GPT-4o 或 DeepSeek
# 如果要测 DeepSeek，就把 api_key 换成 args.api_key (deepseek的key)
# base_url 换成 args.api_url (deepseek的url)

MODEL_NAME = "gpt-4o-mini"  # 或者 "gpt-4o-mini"
API_KEY = args.api_key_gpt
BASE_URL = args.api_url_gpt

# 初始化 LLM
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0, # 测试时温度设为0，保证结果可复现
    api_key=API_KEY,
    base_url=BASE_URL
)

def solve_baseline_question(question_text):
    """
    基准测试逻辑：直接问 LLM，不查库，不推理
    """
    # === C. 构造选项 ===
    options_str = "A: yes\nB: no\nC: maybe"
    
    # --- 构造 Prompt (Zero-shot) ---
    # 注意：这里没有 "Retrieved Evidence" 部分
    final_prompt = f"""
    You are a biomedical expert answering PubMedQA.
    
    Question: {question_text}
    
    Options:
    {options_str}
    
    Instructions:
    1. Decide between yes / no / maybe.
    2. Only output A, B, or C. Nothing else.

    Answer:
    """
    
    try:
        response = llm.invoke(final_prompt)
        answer_text = response.content.strip().replace(".", "")
        # 清洗答案，只取第一个字母
        if len(answer_text) > 0:
            answer_text = answer_text[0].upper()
    except Exception as e:
        print(f"Error: {e}")
        answer_text = "E" # Error

    return answer_text

def run_baseline_benchmark(jsonl_path,
                           num_samples=20,
                           save_path="baseline_pubmedqa_results(4omini).json"):
    # 1. 加载 MedQA 数据
    # === 加载 JSONL ===
    dataset = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line))
    except Exception as e:
        print(f"❌ Failed to load JSONL: {e}")
        return

    print(f">>> Loaded {len(dataset)} samples from {jsonl_path}")
    print(f">>> Start evaluation, saving to {save_path}")

    # PubMedQA 标注映射
    answer_map = {
        "yes": "A",
        "no": "B",
        "maybe": "C"
    }
    results = []
    total = min(num_samples, len(dataset))
    correct = 0

    # 2. 循环测试
    for i in tqdm(range(total), ncols=100, desc="Baseline"):
        item = dataset[i]

        question = item['question']
        gt_raw = item['answer'].lower().strip()
        ground_truth = answer_map.get(gt_raw, "C") # Error if not found
        
        # --- 核心：直接做题 ---
        pred_option = solve_baseline_question(question)
        
        # --- 判分 ---
        is_correct = (pred_option.upper() == ground_truth.upper())
        if is_correct:
            correct += 1
        tqdm.write(f"[Q{i}] Pred={pred_option} | Truth={ground_truth} ({gt_raw})  {'✅' if is_correct else '❌'}")
        # --- 控制台输出 ---
        icon = "✅" if is_correct else "❌"
        # 偶尔打印一下，防止太刷屏
        # tqdm.write(f"[Q{i}] Pred: {pred_option} | Truth: {ground_truth} | {icon}")

        # --- 存入列表 ---
        results.append({
            "id": i,
            "question": question,
            "ground_truth_raw": gt_raw,
            "prediction": pred_option,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "model": MODEL_NAME
        })

        # 实时保存
        if (i + 1) % 10 == 0:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # 3. 最终统计
    accuracy = correct / total * 100
    print("\n" + "="*30)
    print(f"📊 Baseline ({MODEL_NAME}) Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"📂 Details saved to: {os.path.abspath(save_path)}")
    print("="*30)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    testfile_path = "D:/BitLabData/benchmark_data/PubMedQA/PubMedQA_test.json"
    run_baseline_benchmark(
        jsonl_path = testfile_path,   # 修改为你的本地文件路径
        num_samples=240
    )
