# 处理组学表格数据，生成解释型事实知识库
import pandas as pd
import json
from pathlib import Path

# ---------------------------------------------
# 1. 路径配置（根据你的实际路径调整）
# ---------------------------------------------
merged_path = "D:/BitLabData/基因矩阵数据/merged_3tables_fixed.csv"
extra_tables = {
    "mmc7": "D:/BitLabData/基因矩阵数据/需要看懂表格2mmc7.xlsx",
    "mmc8": "D:/BitLabData/基因矩阵数据/需要看懂表格3mmc8.1.xlsx",
    "sample_class": "D:/BitLabData/基因矩阵数据/需要看懂表格5-样本数据分类.xlsx"
}

output_jsonl = "D:/BitLabData/基因矩阵数据/fact_corpus_explanatory.jsonl"

# ---------------------------------------------
# 1. 加载 merged 主表并修复乱码列名
# ---------------------------------------------
try:
    df = pd.read_csv(merged_path, encoding="utf-8")
except:
    df = pd.read_csv(merged_path, encoding="latin1")

rename_map = {
    "¸öÌå±àºÅ": "个体编号",
    "¸öÌåÃû³Æ": "个体名称",
    "ÐÔ±ð": "性别",
    "Ñù±¾±àºÅ": "样本编号",
    "Ñù±¾Ãû³Æ": "样本名称",
    "Ñù±¾ÃèÊöÐÅÏ¢": "样本描述信息",
}
df = df.rename(columns=rename_map)

print("主表列名：", df.columns.tolist())

if "样本名称" not in df.columns:
    raise ValueError("merged CSV 中仍找不到 '样本名称' 列，请检查文件！")

docs = []

# ---------------------------------------------
# 2. 样本级解释型事实
# ---------------------------------------------
def create_sample_fact(sample_name, group):
    patient = group["PatientID"].dropna().iloc[0] if group["PatientID"].notna().any() else "未知患者"
    sample_type = group["SampleType"].dropna().iloc[0] if group["SampleType"].notna().any() else "未知类型"

    # 临床信息提取
    clinical_cols = [
        "Gender","Age","Virus","Stage","AFP_(ng/ml)",
        "CA199_(U/ml)","CEA_(ng/ml)","Pathology","BCLC__stage"
    ]
    clinical_parts = []
    for col in clinical_cols:
        if col in group and group[col].notna().any():
            clinical_parts.append(f"{col}：{group[col].dropna().iloc[0]}")
    clinical_text = "；".join(clinical_parts) if clinical_parts else "临床资料缺失"

    # 基因突变
    genes = sorted(set(group["Hugo_Symbol"].dropna())) if "Hugo_Symbol" in group else []
    if genes:
        gene_list = "、".join(genes[:10])
        mutation_text = f"该样本检测到包括 {gene_list} 在内的多个基因突变。"
    else:
        mutation_text = "此样本未记录明确的突变基因。"

    text = (
        f"样本 {sample_name} 是来自患者 {patient} 的 {sample_type} 类型样本。\n"
        f"患者的临床背景包括：{clinical_text}。\n"
        f"{mutation_text}\n"
        f"这类信息对于理解样本的来源与其在研究中的作用非常重要。"
    )

    return {
        "id": f"sample::{sample_name}",
        "text": text,
        "metadata": {"type": "sample_fact", "sample_name": sample_name, "patient": patient}
    }

# 按样本名称分组
for name, g in df.groupby("样本名称"):
    docs.append(create_sample_fact(name, g))

print(f"已生成样本级事实：{len(docs)} 条")

# ---------------------------------------------
# 3. 附加三表：数据集级解释型事实
# ---------------------------------------------
def create_table_fact(table_name, row):
    parts = []
    for col, val in row.items():
        if pd.isna(val):
            continue
        parts.append(f"{col}：{val}")

    text = (
        f"来自表格 {table_name} 的信息：\n"
        f"{'；'.join(parts)}。\n"
        f"这些内容可以帮助理解该研究中的总体趋势、样本分类或补充特征。"
    )

    return {
        "id": f"{table_name}::{row.name}",
        "text": text,
        "metadata": {"type": "dataset_fact", "table": table_name}
    }

for table_name, path in extra_tables.items():
    try:
        tdf = pd.read_excel(path)
        print(f"加载附加表 {table_name}: {tdf.shape}")
        for idx, row in tdf.iterrows():
            docs.append(create_table_fact(table_name, row))
    except Exception as e:
        print(f"读取 {table_name} 出错：", e)

# ---------------------------------------------
# 4. 写出 JSONL
# ---------------------------------------------
with open(output_jsonl, "w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"\n✨ 已生成完整解释型事实知识库：{output_jsonl}")
print(f"总文档数量：{len(docs)}")