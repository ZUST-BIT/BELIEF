# BELIEF：Structured Evidence Modeling and Uncertainty-Aware Fusion for Biomedical Question Answering

Biomedical Evidence Modeling with Uncertainty-Aware Evidence Fusion

## Project Overview
BELIEF is a high-performance multi-agent biomedical question-answering framework designed specifically for retrieving heterogeneous evidence in biomedical scenarios. It employs a dual-path reasoning architecture, combining neurosemantic reasoning with structured evidence fusion. The multi-agent process is: Question Analysis → Evidence Gathering → Evidence Evaluation → Fusion Decision → Answer Generation. Supported task types include multiple-choice questions (MedQA/MedMCQA) and true/false questions (PubMedQA).

## Directory Structure

```
BELIEF /
├── main.py                  # 主流程示例（多轮检索与推理）
├── agents.py                # 核心智能体实现（A-E + 聚合器）
├── prompt.py                # 主流程提示词模板
├── retriever.py             # PubMed 在线检索逻辑
├── pubmed_online.py         # PubMed API 封装
├── pubmed.py                # 本地 FAISS 检索封装
├── llm_client.py            # LLM 调用封装（API/Ollama/vLLM/Transformers）
├── config.py                # 全局配置（模型、路径、数据库连接）
├── utils/                   # 数据清洗/实体链接等工具
├── faiss_util/              # FAISS 索引构建与检索
├── *_baseline.py            # 基线方法（self_rag/crag/rat 等）
└── requirements.txt
```

## Quick Start

### 1) Install Dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure Parameters

配置文件在 [config.py](config.py) 和 [llm_client.py](llm_client.py)

### 3) Run the Main Pipeline

```bash
python main.py
```
Note: You can customize the question and context targets directly within main.py for ad-hoc testing.

##  Retrieval Modes
Online Retrieval: Queries the live PubMed database via the official API by default (see retriever.py and pubmed_online.py).

Local Retrieval: Performs dense semantic search over a local medical literature pool using FAISS vector indices (see pubmed.py and faiss_util/).

## 许可证

见 [LICENSE](LICENSE)。

## Troubleshooting

### Q1: Index File Not Found or Path Configuration Error
Solution: Rebuild the local database index by running:
```bash
python faiss_util/bio_faiss.py --build
```
### Q2: LLM API Call Failure
Common Causes: Invalid API keys, network connectivity/proxy issues, or exhausted API quotas.
Solution:

  Double-check your endpoints and credentials in config.py.

  Configure an HTTP proxy or swap out the base URL if navigating network restrictions.

  Switch the backend engine to a local model instance (e.g., via Ollama or vLLM).
### Q3: Out of Memory (OOM) Error
Solution:

  Decrease the batch size during FAISS index lookups or model inferences.

  Load models using weight quantization (e.g., int8 or int4).

  Expand your system's virtual swap memory.

## 🤝 Contributing

Contributions, bug reports, and feature requests are highly welcome!

### Contribution Process

1. **Fork** this repository to your own GitHub account.
2. **Create a feature branch**：`git checkout -b feature/awesome-feature`
3. **Commit your changes**：`git commit -m "Add awesome feature"`
4. **Push to the branch**：`git push origin feature/awesome-feature`
5. **Open a Pull Request against our main branch.**

### Code Style

- Adhere strictly to PEP 8 coding standards.
- Document all modules, classes, and methods with descriptive docstrings.
- Run existing test suites before submitting your PR: `pytest tests/`

---

## 🛠️ Tech Stack

| Component | Technology |
|------|------|
| **LLM** | GPT-4o-mini / GPT-4o / DeepSeek / Qwen |
| **Literature Database** | PubMed |
| **Core Frameworks** | LangChain + PyTorch |
| **NLP Utilities** | spaCy + SciSpaCy |
|**Inference Engine** | vLLM |

---

## 📄 License

This project is licensed under the terms of the MIT License.

```
MIT License

Copyright (c) 2026 BitLab Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 📧 Contact

For questions, suggestions, or potential collaborations, please reach out via:

- **GitHub Issues**: Submit an issue directly to our repository tracker.
- **Email**: ninghao@zust.edu.cn
- **Project Homepage**: ...

---


<div align="center">

**⭐ 如果本项目对您有帮助，请给个 Star！⭐**

Made with ❤️ by BitLab Research Team

</div>
