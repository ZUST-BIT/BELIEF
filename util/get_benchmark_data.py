# 获取 MedAgentsBench 基准数据集的脚本
import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# 定义所有子数据集的名称
SUBSETS = [
    "MedQA",         # 美国医师执照考试 (USMLE) - 必测
    "PubMedQA",      # 基于文献的推理 (Yes/No/Maybe) - 必测
    "MedMCQA",       # 印度医学入学考试 - 题量大
    "MMLU",          # 大模型通用能力的医学部分 - 基准
    "MMLU-Pro",      # MMLU 的升级版 - 更难
    "MedBullets",    # 临床要点问答 - 偏向基础知识
    "AfrimedQA",     # 非洲医疗背景 - 偏向特定地域
    "MedExQA",       # 专家级问答
    "MedXpertQA-R",  # 专家级推理 (Reasoning) - 重点
    "MedXpertQA-U",  # 专家级理解 (Understanding) - 重点
]

# 保存路径
SAVE_DIR = "benchmark_data"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def download_and_save(subset_name):
    print(f"\n🚀 正在处理: {subset_name} ...")
    
    # 我们尝试下载两个 split: 'test' 和 'test_hard'
    splits_to_try = ["test", "test_hard"]
    
    for split in splits_to_try:
        try:
            # 1. 加载数据
            dataset = load_dataset(
                "super-dainiu/medagents-benchmark", 
                subset_name, 
                split=split, 
                trust_remote_code=True
            )
            
            # 2. 转为 DataFrame
            df = pd.DataFrame(dataset)
            
            # 3. 保存为 JSON (保留原始结构，推荐看这个)
            json_filename = f"{subset_name}_{split}.json"
            json_path = os.path.join(SAVE_DIR, json_filename)
            df.to_json(json_path, orient="records", lines=True, force_ascii=False)
            
            # 4. 保存为 Excel (方便人工抽查)
            # excel_filename = f"{subset_name}_{split}.xlsx"
            # excel_path = os.path.join(SAVE_DIR, excel_filename)
            # df.to_excel(excel_path, index=False)
            
            print(f"  ✅ [成功] {split}: {len(df)} 条 -> {json_filename}")
            
        except Exception as e:
            # 有些数据集可能没有 test_hard，这是正常的
            if "Split" in str(e) or "Unknown split" in str(e):
                print(f"  ⚠️ [跳过] {split}: 该数据集不包含此划分")
            else:
                print(f"  ❌ [失败] {split}: {str(e)[:100]}")

if __name__ == "__main__":
    print("开始全量下载 MedAgentsBench...")
    for subset in tqdm(SUBSETS):
        download_and_save(subset)
    print(f"\n🎉 所有下载完成！文件保存在: {os.path.abspath(SAVE_DIR)}")