# MEDAR-QA

**M**ultimodal **E**vidence **D**iscovery **a**nd **R**easoning for Biomedical **Q**uestion **A**nswering

一个基于多源证据检索与迭代推理的生物医学问答系统，支持多智能体协作和证据理论的不确定性推理。

---

## 📖 项目简介

MEDAR-QA 是一个面向复杂生物医学问题的智能问答系统，通过整合 **PrimeKG 知识图谱**、**PubMed 文献检索** 和 **大语言模型（LLM）**，实现了基于证据理论（Dempster-Shafer Theory）的多轮迭代推理框架。系统能够自动判断证据充分性，并在需要时触发额外检索，确保答案的可靠性和可解释性。

### 核心特性

- 🔍 **混合检索架构**：融合 PubMed 文献向量检索、Neo4j 知识图谱查询和实体对齐机制
- 🤖 **多智能体协作**：包含意图路由、多维度证据分析、全局评估和答案生成等专业智能体
- 🔄 **自适应迭代推理**：基于 D-S 理论的证据充分性评估，动态触发多轮检索-评估循环
- 📊 **完整可解释性**：记录每轮检索的证据来源、置信度计算和决策过程
- 🎯 **多任务支持**：支持 Yes/No 问答（PubMedQA）和多项选择题（MedQA）等多种任务格式
- 🧩 **模块化设计**：各组件松耦合，便于扩展新的检索源或智能体

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户问题输入                              │
│                    (Question + Optional Context)                 │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agent A: Intent Router                        │
│         • 问题类型识别（定义查询/因果推理/治疗方案等）              │
│         • 生物医学实体提取（疾病/药物/基因/通路等）                 │
│         • PICO 框架分析（针对临床问题）                            │
│         • 检索策略规划（确定需要查询的知识源）                      │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
                 ┌────────────────────────┐
                 │   是否需要检索？        │
                 │  （基于问题复杂度）      │
                 └───┬────────────────┬───┘
                     │ No (简单问题)   │ Yes
                     ▼                ▼
              直接生成答案    ┌─────────────────────────────────┐
                             │   Hybrid Retrieval Process      │
                             │  ┌───────────────────────────┐  │
                             │  │  Entity Linker            │  │
                             │  │  • 字典匹配 + 向量检索      │  │
                             │  │  • MeSH 术语标准化         │  │
                             │  └───────────────────────────┘  │
                             │  ┌───────────────────────────┐  │
                             │  │  Neo4j Knowledge Graph    │  │
                             │  │  • PrimeKG 子图检索        │  │
                             │  │  • 多跳关系推理           │  │
                             │  └───────────────────────────┘  │
                             │  ┌───────────────────────────┐  │
                             │  │  PubMed Vector Search     │  │
                             │  │  • FAISS 向量检索          │  │
                             │  │  • MongoDB 全文检索        │  │
                             │  └───────────────────────────┘  │
                             └──────────────┬──────────────────┘
                                            ▼
                ┌─────────────────────────────────────────────────┐
                │         Agent B: MuldimAnalyst                   │
                │  • 实体对齐验证（检索结果与问题实体的匹配度）       │
                │  • 意图匹配评估（证据与问题类型的相关性）          │
                │  • 信息密度分析（证据的信息量和完整性）           │
                │  • 证据精炼与打分（去重、排序、标注来源）          │
                └──────────────────────┬──────────────────────────┘
                                       ▼
                ┌─────────────────────────────────────────────────┐
                │      Agent C: EvidenceEvaluator (D-S Theory)     │
                │  • 计算每条证据的置信度（Belief）                 │
                │  • 不确定性间隙分析（Uncertainty Gap）            │
                │  • 证据冲突检测（Conflict Score）                │
                │  • GO/NO-GO 决策（是否需要额外检索）              │
                │  • 生成改进策略（新的检索 query）                 │
                └──────────────────────┬──────────────────────────┘
                                       ▼
                      ┌────────────────────────────┐
                      │   Decision Controller       │
                      │  （完整性控制器）            │
                      └───┬────────────────────┬───┘
                          │ NO-GO              │ GO
                          │ (需要更多证据)      │ (证据充足)
                          ▼                    ▼
                    重新检索          ┌─────────────────────────┐
                 (迭代 ≤ 最大轮数)     │  Agent E: Generator     │
                          │           │  • 综合所有精炼证据      │
                          └───────────│  • 生成结构化答案       │
                                      │  • 提供推理依据         │
                                      └─────────────────────────┘
```

---

## 📁 项目结构

```
MEDAR-QA/
├── main.py                   # 主程序入口，完整的多轮迭代推理流程
├── run.py                    # 简化版运行脚本
├── config.py                 # 系统配置参数（数据库连接、API 密钥、模型路径）
│
├── agents.py                 # 五大智能体实现
│   ├── AgentA (IntentAnalyzer)        # 意图分析与检索规划
│   ├── AgentB (MuldimAnalyst)         # 多维度证据分析
│   ├── AgentC (EvidenceEvaluator)     # 证据评估（D-S Theory）
│   ├── AgentD (CompletenessController)# 完整性控制器
│   └── AgentE (Generator)             # 答案生成器
│
├── intent_router.py          # 意图路由模块（预判是否需要检索）
├── retriever.py              # 混合检索模块（整合多源知识）
├── context_manager.py        # 上下文管理器（证据池与工作流日志）
├── llm_client.py             # LLM 客户端封装（支持 OpenAI/DeepSeek）
├── prompt.py                 # 各智能体的 Prompt 模板定义
│
├── neo4j.py                  # Neo4j 知识图谱交互模块
├── pubmed.py                 # PubMed 文献检索模块（MongoDB + FAISS）
├── omic.py                   # 组学数据检索模块
├── agentic_tools.py          # 智能体辅助工具函数
│
├── test_medqa_baseline.py    # MedQA 基线测试（无检索）
├── test_medqa_rag.py         # MedQA RAG 测试（单轮检索）
├── test_medqa.py             # MedQA 完整测试（MEDAR-QA 框架）
├── test_pubmedqa_rag.py      # PubMedQA RAG 测试
├── test_pubmedqa_with_context.py  # PubMedQA 带 Context 测试
├── test_pubmedqa.py          # PubMedQA 基线测试
│
├── requirements.txt          # Python 依赖
│
├── utils/                    # 工具模块
│   ├── entity_linker.py      # 实体链接（字典+向量混合检索）
│   └── data_refiner.py       # 证据数据清洗与格式化
│
├── faiss_util/               # FAISS 向量检索工具
│   ├── bio_faiss.py          # 生物医学文献向量检索
│   ├── hcc_faiss.py          # 肝癌（HCC）相关文献检索
│   └── omic_faiss.py         # 组学数据向量检索
│
├── kg_index_output/          # 知识图谱索引文件
│   ├── keyword_mapping.json  # 实体名称映射表
│   └── primekg_mesh.index    # PrimeKG-MeSH 向量索引
│
├── output/                   # 输出文件
│   ├── faiss_index_neo4j.index          # Neo4j 实体向量索引
│   ├── faiss_index_primekg_sapbert.index # PrimeKG SapBERT 索引
│   ├── faiss_mapping_bio.json           # 生物医学文献映射
│   ├── faiss_mapping_hcc.json           # 肝癌文献映射
│   └── faiss_mapping_omic.json          # 组学数据映射
│
├── data/                     # 测试数据集
│   ├── medqa_sample.jsonl               # MedQA 样本数据
│   ├── medqa_with_context_sample.json   # 带 Context 的 MedQA 数据
│   └── pubmedqa_sample.json             # PubMedQA 样本数据
│
├── TEST_RESULTS/             # 测试结果输出
│   ├── medqa/                           # MedQA 测试结果
│   └── pubmedqa/                        # PubMedQA 测试结果
│
└── kg_index_output/          # 知识图谱索引文件
    ├── keyword_mapping.json             # 实体名称映射表
    └── primekg_mesh.index               # PrimeKG-MeSH 向量索引
```

---

## 🚀 快速开始

### 1. 环境要求

- **Python**: 3.9+
- **Neo4j**: 4.0+（存储 PrimeKG 知识图谱）
- **MongoDB**: 4.0+（可选，存储 PubMed 文献）
- **GPU**: 推荐使用 CUDA 加速向量检索和模型推理

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-repo/MEDAR-QA.git
cd MEDAR-QA

# 安装依赖
pip install -r requirements.txt

# 下载 spaCy 模型
python -m spacy download en_core_web_sm
python -m spacy download en_core_sci_sm
```

### 3. 数据准备

#### 3.1 Neo4j 知识图谱设置

1. 下载并导入 [PrimeKG](https://github.com/mims-harvard/PrimeKG) 数据到 Neo4j
2. 确保图谱包含以下节点类型：
   - `GeneProtein`, `Drug`, `Disease`, `Pathway`, `BiologicalProcess` 等

#### 3.2 FAISS 索引构建（可选）

如果需要使用 PubMed 文献检索，需要构建 FAISS 索引：

```bash
# 构建生物医学文献索引
python faiss_util/bio_faiss.py --build

# 构建组学数据索引
python faiss_util/omic_faiss.py --build
```

### 4. 配置参数

编辑 [config.py](config.py) 中的配置：

```python
# Neo4j 配置
--neo4j_url='bolt://localhost:7687'
--neo4j_usr='neo4j'
--neo4j_pwd='your_password'

# MongoDB 配置（可选）
--mongo_url='mongodb://localhost:27017/'
--db_name='bio'
--collection_name='pubmed'

# LLM API 配置
--api_url_gpt='https://api.openai.com/v1'
--api_key_gpt='your_api_key'

# 嵌入模型路径
--embedding_model_en='BAAI/bge-large-en-v1.5'

# MeSH 数据路径
--mesh_mapping_path='path/to/mesh_mapping.json'
--mesh_info_path='path/to/mesh_info.json'
```

### 5. 运行系统

#### 5.1 交互式问答

```bash
python main.py
```

在代码中修改问题示例：

```python
# 在 main.py 中修改测试问题
question = "What is the mechanism of Osimertinib resistance in NSCLC?"
context = ""  # 可选的背景信息
```

#### 5.2 批量测试

**测试 PubMedQA 数据集：**

```bash
# 完整 MEDAR-QA 框架测试
python test_pubmedqa_with_context.py

# RAG 基线测试
python test_pubmedqa_rag.py

# 无检索基线测试
python test_pubmedqa.py
```

**测试 MedQA 数据集：**

```bash
# 完整 MEDAR-QA 框架测试
python test_medqa.py

# RAG 基线测试
python test_medqa_rag.py

# 无检索基线测试
python test_medqa_baseline.py
```

---

## 🔧 核心模块详解

### 1. Agent A: IntentAnalyzer（意图分析器）

**功能：**
- 问题类型识别（定义查询、因果推理、治疗方案、诊断推理等）
- 生物医学实体提取（疾病、药物、基因、蛋白、通路等）
- PICO 框架分解（针对临床问题）
- 检索策略生成

**输出示例：**
```json
{
  "question_type": "mechanism_query",
  "biomedical_entities": ["Osimertinib", "NSCLC", "EGFR"],
  "search_queries": [
    "Osimertinib resistance mechanisms in NSCLC",
    "EGFR T790M mutation"
  ]
}
```

### 2. Hybrid Retriever（混合检索器）

**检索流程：**

1. **实体链接**（`utils/entity_linker.py`）
   - 字典匹配：快速查找标准化实体
   - SapBERT 向量匹配：处理非标准表述

2. **知识图谱检索**（`neo4j.py`）
   - 子图提取：以实体为中心的 2-3 跳关系
   - 关系推理：疾病-药物-靶点-通路关联

3. **文献检索**（`pubmed.py`）
   - FAISS 向量检索：基于语义相似度
   - MongoDB 全文检索：关键词匹配
   - BM25 排序：混合评分

### 3. Agent B: MuldimAnalyst（多维度分析器）

**评估维度：**
- **实体对齐度** (0-1)：证据中包含的问题实体数量占比
- **意图匹配度** (0-1)：证据类型与问题类型的相关性
- **信息密度** (0-1)：证据的信息量和完整性

**输出示例：**
```json
{
  "entity_alignment": 0.85,
  "intent_match": 0.92,
  "information_density": 0.78,
  "overall_score": 0.85,
  "refined_evidence": "精炼后的证据文本..."
}
```

### 4. Agent C: EvidenceEvaluator（证据评估器）

**基于 Dempster-Shafer 理论：**

- **Belief (Bel)**：支持某假设的证据强度
- **Plausibility (Pl)**：不排斥某假设的可能性
- **Uncertainty (U = Pl - Bel)**：不确定性间隙

**决策逻辑：**
```
if Bel ≥ 0.7 and U < 0.2:
    decision = "GO"  # 证据充足，生成答案
else:
    decision = "NO-GO"  # 需要更多证据
    generate_improvement_strategy()
```

### 5. Agent D: CompletenessController（完整性控制器）

**职责：**
- 监控迭代次数（默认最大 3 轮）
- 检测证据池变化（新增证据数量）
- 判断是否触发下一轮检索

### 6. Agent E: Generator（答案生成器）

**特性：**
- 多任务支持：Yes/No 问答、多项选择题、开放式问答
- 可解释性：引用具体证据来源
- 格式化输出：JSON 结构化答案

---

## 📊 评估与实验

### 支持的数据集

| 数据集 | 任务类型 | 样本数量 | 数据文件 |
|--------|---------|---------|---------|
| **PubMedQA** | Yes/No/Maybe 问答 | 1000 | `data/pubmedqa_sample.json` |
| **MedQA** | 四选一多选题 | 1273 | `data/medqa_sample.jsonl` |

### 评估指标

- **准确率（Accuracy）**：正确答案比例
- **证据充分性（Evidence Sufficiency）**：平均置信度
- **检索效率（Retrieval Efficiency）**：平均检索轮数

### 实验结果示例

```
Dataset: PubMedQA
Total Questions: 500
Accuracy: 72.4%
Avg Iterations: 1.8
Avg Confidence: 0.81
```

结果文件保存在 `TEST_RESULTS/` 目录下，包含：
- 完整的问题-答案对
- 每题的证据来源
- 置信度和迭代轮数

---

## 🔬 技术亮点

### 1. 自适应迭代推理

系统通过 D-S 理论动态评估证据充分性，避免过度检索或信息不足：

```python
# 伪代码示例
while iteration < max_iterations:
    evidence = retrieve(query)
    scores = analyze_evidence(evidence)
    decision = evaluate_sufficiency(scores)
    
    if decision == "GO":
        answer = generate(evidence)
        break
    else:
        query = improve_query(decision.feedback)
```

### 2. 多源知识融合

- **结构化知识**：PrimeKG 知识图谱（180万+ 三元组）
- **非结构化知识**：PubMed 文献摘要（3000万+ 篇）
- **实体标准化**：MeSH 术语体系（28万+ 概念）

### 3. 证据理论（Dempster-Shafer Theory）

优势：
- 显式建模不确定性（与概率论不同）
- 支持证据冲突检测
- 量化信息完整性

### 4. 模块化与可扩展性

- 每个智能体可独立替换或升级
- 支持添加新的检索源（如临床指南、基因组数据库）
- Prompt 模板集中管理，易于调优

---

## 📖 使用示例

### 示例 1：机制查询

**问题：** *What is the mechanism of cisplatin-induced ototoxicity?*

**系统输出：**
```json
{
  "answer": "Cisplatin causes ototoxicity primarily through ROS generation in cochlear cells...",
  "evidence": [
    {
      "source": "PubMed",
      "pmid": "12345678",
      "text": "Cisplatin generates ROS in cochlear cells leading to apoptosis...",
      "confidence": 0.89
    },
    {
      "source": "PrimeKG",
      "relation": "Cisplatin -> causes -> Hearing Loss",
      "confidence": 0.75
    }
  ],
  "iterations": 2,
  "final_confidence": 0.84
}
```

### 示例 2：临床问答

**问题：** *Syncope during bathing in infants, a pediatric form of water-induced urticaria?*

**系统输出：**
```json
{
  "answer": "yes",
  "reasoning": "Based on evidence showing histamine increase after water exposure and family history of dermographism...",
  "evidence_count": 8,
  "iterations": 1
}
```

---

## 📝 工作流日志

系统会自动记录每次问答的完整推理过程，方便调试和分析：

### 日志文件

- **工作流报告**：记录每轮检索的决策和证据
- **证据池快照**：保存所有检索到的证据
- **评估结果**：D-S 理论计算的置信度和不确定性

### 日志内容示例

```
=== 第 1 轮检索 ===
检索查询: syncope bathing infants water-induced urticaria
检索来源: PubMed (8), PrimeKG (3)

=== Agent B 分析 ===
平均实体对齐: 0.82
平均意图匹配: 0.91
平均信息密度: 0.76

=== Agent C 评估 ===
总体置信度 (Bel): 0.83
不确定性 (U): 0.12
决策: GO → 证据充足，生成答案
```

---

## 🛠️ 高级配置

### 自定义智能体

在 [agents.py](agents.py) 中添加新的智能体：

```python
class CustomAgent:
    def __init__(self):
        self.llm = get_llm_client()
    
    def process(self, input_data):
        prompt = f"Your custom prompt: {input_data}"
        response = self.llm.chat(prompt)
        return response
```

### 调整检索参数

在 [retriever.py](retriever.py) 中配置：

```python
# 向量检索 Top-K
FAISS_TOP_K = 10

# 知识图谱跳数
KG_HOP_LIMIT = 2

# 文献检索来源权重
PUBMED_WEIGHT = 0.6
KG_WEIGHT = 0.4
```

### 修改评估阈值

在 [agents.py](agents.py) 的 `AgentC` 中调整：

```python
# 置信度阈值（GO 决策）
CONFIDENCE_THRESHOLD = 0.7

# 不确定性阈值
UNCERTAINTY_THRESHOLD = 0.2

# 最大迭代次数
MAX_ITERATIONS = 3
```

---

## 🐛 常见问题

### Q1: Neo4j 连接失败

**原因：** 数据库未启动或连接配置错误

**解决方案：**
1. 确认 Neo4j 服务已启动：`neo4j status`
2. 检查 [config.py](config.py) 中的连接配置
3. 测试连接：
   ```bash
   python -c "from neo4j import Neo4jManager; Neo4jManager().test_connection()"
   ```

### Q2: FAISS 索引加载错误

**原因：** 索引文件不存在或路径配置错误

**解决方案：**
```bash
# 重新构建索引
python faiss_util/bio_faiss.py --build

# 检查索引文件是否存在
ls output/faiss_index_bio.bin
```

### Q3: LLM API 调用失败

**常见原因：**
- API 密钥错误
- 网络连接问题
- API 调用额度不足

**解决方案：**
- 检查 `config.py` 中的 API 配置
- 使用代理或更换 API 地址
- 切换到本地模型（如 Ollama）

### Q4: 内存不足错误

**解决方案：**
- 减小 FAISS 索引的批处理大小
- 使用量化模型（如 int8）
- 增加系统虚拟内存

---

## 🚧 开发计划

### 即将推出

- [ ] 支持多语言问答（中文医学问答）
- [ ] 集成更多知识源（DrugBank、ClinicalTrials.gov、KEGG）
- [ ] 添加用户反馈机制（在线学习）
- [ ] 开发 Web UI 界面（Gradio/Streamlit）

### 长期规划

- [ ] 支持多模态输入（医学影像、病理切片、基因序列）
- [ ] 实现联邦学习框架（保护隐私）
- [ ] 构建临床决策支持系统
- [ ] 发布预训练的领域模型

---

## 🤝 贡献指南

欢迎贡献代码、提出问题或建议！

### 贡献流程

1. **Fork** 本项目到你的 GitHub 账号
2. **创建特性分支**：`git checkout -b feature/awesome-feature`
3. **提交更改**：`git commit -m "Add awesome feature"`
4. **推送到分支**：`git push origin feature/awesome-feature`
5. **提交 Pull Request**

### 代码规范

- 遵循 PEP 8 编码规范
- 添加必要的注释和文档字符串
- 提交前运行测试：`pytest tests/`

---

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| **LLM** | GPT-4o-mini / GPT-4o / DeepSeek / Qwen |
| **向量检索** | FAISS + SapBERT / BGE-large |
| **知识图谱** | Neo4j + PrimeKG |
| **文献数据** | PubMed / MongoDB |
| **框架** | LangChain + PyTorch |
| **NLP 工具** | spaCy + SciSpaCy |

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

```
MIT License

Copyright (c) 2026 BitLab Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 📧 联系方式

如有问题、建议或合作意向，欢迎联系：

- **GitHub Issues**: [提交问题](https://github.com/your-repo/MEDAR-QA/issues)
- **Email**: your-email@example.com
- **项目主页**: [MEDAR-QA Documentation](https://your-docs-site.com)

---

## 🙏 致谢

本项目依赖以下优秀的开源项目和数据资源：

- **[PrimeKG](https://github.com/mims-harvard/PrimeKG)**: Harvard Medical School 构建的精准医疗知识图谱
- **[PubMedQA](https://pubmedqa.github.io/)**: 生物医学文献问答基准数据集
- **[MedQA](https://github.com/jind11/MedQA)**: 医学考试问答数据集
- **[SapBERT](https://github.com/cambridgeltl/sapbert)**: Cambridge 的生物医学实体嵌入模型
- **[FAISS](https://github.com/facebookresearch/faiss)**: Meta 的高效向量检索库
- **[LangChain](https://github.com/langchain-ai/langchain)**: LLM 应用开发框架

---

## 📚 参考文献

1. Chandak, P., et al. (2023). "Building a knowledge graph to enable precision medicine". *Nature Scientific Data*, 10, 67.
2. Jin, Q., et al. (2019). "PubMedQA: A dataset for biomedical research question answering". *EMNLP*.
3. Shafer, G. (1976). "A Mathematical Theory of Evidence". *Princeton University Press*.
4. Liu, J., et al. (2021). "Self-Alignment Pretraining for Biomedical Entity Representations". *NAACL*.
5. Johnson, J., et al. (2019). "Billion-scale similarity search with GPUs". *IEEE Transactions on Big Data*.

---

<div align="center">

**⭐ 如果本项目对您有帮助，请给个 Star！⭐**

Made with ❤️ by BitLab Research Team

[文档](https://docs.example.com) • [论文](https://arxiv.org/xxx) • [演示视频](https://youtube.com/xxx)

</div>
