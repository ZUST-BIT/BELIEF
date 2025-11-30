import pandas as pd
import re

# ---------------------------------------------
# Step 1: Load all Excel files
# ---------------------------------------------
paths = {
    "mmc4": "D:/BitLabData/基因矩阵数据/需要看懂表格1-mmc4.xlsx",   # Mutation table
    "mmc7": "D:/BitLabData/基因矩阵数据/需要看懂表格2mmc7.xlsx",
    "mmc8": "D:/BitLabData/基因矩阵数据/需要看懂表格3mmc8.xlsx",
    "sample_class": "D:/BitLabData/基因矩阵数据/需要看懂表格5-样本数据分类.xlsx",
    "clinical": "D:/BitLabData/基因矩阵数据/需要看懂表格6-able S1. Patients' clinicopathological i.xlsx",
    "sample_info": "D:/BitLabData/基因矩阵数据/需要看懂表格7-个体编号样本编号信息.xlsx",
}

mut = pd.read_excel(paths["mmc4"])
sample_raw = pd.read_excel(paths["sample_info"], header=None)
clinical = pd.read_excel(paths["clinical"])

# -------------------------------------------------------
# Step 2. Fix Sample Info Table (关键修复在这里)
# -------------------------------------------------------
sample_info = sample_raw.copy()

# 第2行是表头
sample_info.columns = sample_info.iloc[1].tolist()

# 从第3行开始为数据
sample_info = sample_info.iloc[2:].reset_index(drop=True)

# 只保留需要的列
sample_info = sample_info[["个体编号", "个体名称", "性别", "样本编号", "样本名称", "样本描述信息"]]

# ⚠⚠⚠ 关键修复：填充合并单元导致的空白
sample_info["个体编号"] = sample_info["个体编号"].ffill()
sample_info["个体名称"] = sample_info["个体名称"].ffill()

# PatientID = 个体名称 (如 P1)
sample_info["PatientID"] = sample_info["个体名称"].astype(str).str.strip()

# PDO ID
sample_info["PDO_ID"] = sample_info["样本名称"].str.extract(r"^(P\\d+C\\d+)")

# 样本类型
def map_sample_type(desc):
    if isinstance(desc, str):
        d = desc.lower()
        if "organoid" in d:
            return "Organoid"
        if "tumor" in d:
            return "Tumor"
        if "blood" in d:
            return "Blood"
    return "Unknown"

sample_info["SampleType"] = sample_info["样本描述信息"].apply(map_sample_type)

# -------------------------------------------------------
# Step 3. Clean Clinical Table
# -------------------------------------------------------
clinical.columns = clinical.columns.str.strip().str.replace(" ", "_")

clinical["Patient_number"] = clinical["Patient_number"].astype(str).str.strip()

# -------------------------------------------------------
# Step 4. Merge Mutation + Sample Info
# -------------------------------------------------------
merged = mut.merge(
    sample_info,
    how="left",
    left_on="Tumor_Sample_Barcode",
    right_on="样本名称"
)

# -------------------------------------------------------
# Step 5. Merge with Clinical Info
# -------------------------------------------------------
merged = merged.merge(
    clinical,
    how="left",
    left_on="PatientID",
    right_on="Patient_number"
)

# -------------------------------------------------------
# Step 6. Save
# -------------------------------------------------------
output_path = "D:/BitLabData/基因矩阵数据/merged_3tables_fixed.csv"
merged.to_csv(output_path, index=False)

print("修复后合并完成！")
print("输出文件：", output_path)
print("尺寸：", merged.shape)