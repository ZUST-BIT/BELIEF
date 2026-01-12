# MEDAR-QA

**M**ultimodal **E**vidence **D**iscovery **a**nd **R**easoning for Biomedical **Q**uestion **A**nswering

一个基于多源证据检索与迭代推理的生物医学问答系统。

---

## 📖 项目简介

MEDAR-QA 是一个智能医学问答系统，结合了 **知识图谱（PrimeKG）**、**PubMed 文献检索** 和 **大语言模型（LLM）** 的能力，通过多轮迭代检索与证据评估机制，为复杂的生物医学问题提供可靠、可解释的答案。

### 核心特性

- 🔍 **多源证据检索**：整合 PubMed 文献、PrimeKG 知识图谱和实体定义等多种知识源
- 🤖 **多智能体协作**：包含意图路由、证据分析、全局评估和答案生成等多个专业智能体
- 🔄 **迭代式推理**：支持多轮检索-评估循环，确保证据充分性
- 📊 **可解释性报告**：提供完整的工作流日志和推理过程记录
- 🎯 **基于 Dempster-Shafer 理论**：使用证据理论进行不确定性量化和冲突检测

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户问题输入                              │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Intent Router (意图路由)                       │
│         - 问题分析与实体提取                                       │
│         - 检索计划生成                                            │
│         - Query 重写优化                                          │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Hybrid Retriever (混合检索)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Entity Linker │  │  Neo4j KG    │  │  PubMed Literature   │   │
│  │ (实体对齐)     │  │  (知识图谱)   │  │  (文献向量检索)        │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 MuldimAnalyst (多维度分析器)                       │
│         - 实体对齐检查                                            │
│         - 意图匹配验证                                            │
│         - 信息密度评估                                            │
│         - 证据精炼与打分                                          │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               EvidenceEvaluator (证据评估器)                      │
│         - D-S 理论置信度计算                                      │
│         - 不确定性间隙分析                                        │
│         - GO/NO-GO 决策                                          │
│         - 改进策略生成                                            │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
          ┌──────────────────────────────────────┐
          │            决策分支                   │
          │   GO → 证据充足，生成答案              │
          │   NO-GO → 继续迭代检索                │
          └──────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                GeneratorAgent (答案生成器)                        │
│         - 基于精炼证据生成最终答案                                 │
│         - 支持 PubMedQA 等任务格式                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
MEDAR-QA/
├── run.py                    # 主程序入口，问答系统核心流程
├── config.py                 # 系统配置参数（数据库连接、API密钥等）
├── agentic_tools.py          # 智能体工具（分析器、评估器、生成器）
├── intent_router.py          # 意图路由模块（问题分析与检索规划）
├── retriever.py              # 混合检索模块（整合多源知识）
├── context_manager.py        # 上下文管理器（证据池与工作流日志）
├── neo4j.py                  # Neo4j 知识图谱交互模块
├── pubmed.py                 # PubMed 文献检索模块
├── prompt.py                 # Prompt 模板定义
├── query.py                  # 查询配置
├── evaluate_pubmedqa.py      # PubMedQA 数据集评估脚本
├── requirements.txt          # Python 依赖
│
├── utils/                    # 工具模块
│   ├── entity_linker.py      # 实体链接（字典+向量混合检索）
│   └── data_refiner.py       # 数据清洗与格式化
│
├── faiss_util/               # FAISS 向量检索工具
│   ├── bio_faiss.py          # 生物医学文献向量检索
│   ├── hcc_faiss.py          # HCC 相关文献检索
│   └── omic_faiss.py         # 组学数据检索
│
├── kg_index_output/          # 知识图谱索引文件
│   ├── keyword_mapping.json  # 实体名称映射表
│   └── primekg_mesh.index    # PrimeKG-MeSH 向量索引
│
├── output/                   # 输出文件
│   ├── faiss_index_*.index   # FAISS 索引文件
│   └── faiss_mapping_*.json  # FAISS ID 映射
│
├── results/                  # 评估结果
│   └── evaluation_pubmedqa/  # PubMedQA 评估日志
│
├── workflow_logs/            # 工作流日志（JSON + TXT）
└── exper_script/             # 实验脚本
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.9+
- Neo4j 数据库（存储 PrimeKG 知识图谱）
- MongoDB（可选，存储 PubMed 文献）
- CUDA（可选，加速向量检索）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置参数

修改 [config.py](config.py) 中的配置参数：

```python
# Neo4j 数据库配置
--neo4j_url     # Neo4j 连接地址
--neo4j_usr     # 用户名
--neo4j_pwd     # 密码

# OpenAI API 配置
--api_url_gpt   # API 地址
--api_key_gpt   # API 密钥

# 数据路径配置
--mesh_mapping_path  # MeSH 映射文件路径
--mesh_info_path     # MeSH 信息文件路径
```

### 4. 运行系统

```bash
python run.py
```

或在代码中调用：

```python
from run import run_qa_system

question = "What is the mechanism of resistance to Osimertinib in NSCLC?"
answer = run_qa_system(question)
print(answer)
```

---

## 🔧 核心模块说明

### IntentRouter（意图路由）

- 分析用户问题，提取关键实体
- 生成优化的检索查询
- 规划检索策略（文献/知识图谱）

### EntityRetriever（实体链接器）

- **字典精确匹配**：基于预构建的关键词映射表
- **向量模糊匹配**：使用 SapBERT 进行语义相似度检索
- 将问题中的实体对齐到 PrimeKG 知识图谱

### MuldimAnalyst（多维度分析器）

- 逐条评估检索到的证据
- 执行实体对齐、意图匹配、信息密度评估
- 输出相关性评分和精炼后的证据

### EvidenceEvaluator（证据评估器）

- 基于 Dempster-Shafer 理论计算证据置信度
- 检测证据间的冲突
- 做出 GO/NO-GO 决策，指导迭代检索

### GeneratorAgent（答案生成器）

- 基于精炼后的证据生成最终答案
- 支持多种任务格式（如 PubMedQA）

### MultimodalContext（上下文管理器）

- 管理证据池，去重与汇总
- 记录完整的工作流历史
- 生成可解释性报告

---

## 📊 评估与基准测试

### PubMedQA 评估

```bash
python evaluate_pubmedqa.py
```

评估结果将保存至 `results/evaluation_pubmedqa/` 目录。

### 已支持的数据集

- [PubMedQA](https://pubmedqa.github.io/) - 生物医学文献问答
- MedQA - 医学考试问答

---

## 📝 工作流日志

系统会自动保存每次问答的工作流日志：

- **JSON 格式**：`workflow_logs/workflow_log_<timestamp>.json`
- **可读报告**：`workflow_logs/workflow_log_<timestamp>.txt`

日志内容包括：
- 每轮检索的查询与结果
- 证据分析与评分详情
- 评估器决策与改进策略
- 最终答案

---

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| LLM | GPT-4o-mini / GPT-4o / DeepSeek |
| 向量检索 | FAISS + SapBERT / BGE |
| 知识图谱 | Neo4j + PrimeKG |
| 文献数据 | PubMed / MongoDB |
| 框架 | LangChain |

---

## 📄 License

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 📧 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。
