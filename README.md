# BELIEF：Structured Evidence Modeling and Uncertainty-Aware Fusion for Biomedical Question Answering

Biomedical Evidence Modeling with Uncertainty-Aware Evidence Fusion

## 项目概览

- 多智能体流水线：问题分析、证据整理、证据评估、融合决策、答案生成
- 支持选择题（MedQA/MedMCQA）与是非题（PubMedQA）等任务

## 目录结构

```
MEDAR-QA/
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

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 配置关键参数

配置文件在 [config.py](config.py) 和 [llm_client.py](llm_client.py)

### 3) 运行主流程

```bash
python main.py
```

在 [main.py](main.py) 中可以直接替换 `question` 与 `context`。

## 检索方式说明

- 在线检索：默认走 PubMed Online（见 [retriever.py](retriever.py) + [pubmed_online.py](pubmed_online.py)）
- 本地检索：FAISS + 本地索引（见 [pubmed.py](pubmed.py) 与 [faiss_util/](faiss_util/)）

## 许可证

见 [LICENSE](LICENSE)。

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
| **文献数据** | PubMed |
| **框架** | LangChain + PyTorch |
| **NLP 工具** | spaCy + SciSpaCy |
|**LLM 推理框架** | vLLM |

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
- **Email**: ninghao@zust.edu.cn
- **项目主页**: ...

---


<div align="center">

**⭐ 如果本项目对您有帮助，请给个 Star！⭐**

Made with ❤️ by BitLab Research Team

</div>
