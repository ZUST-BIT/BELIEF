# BELIEF
Structured Evidence Modeling and Uncertainty-Aware Fusion for Biomedical Question Answering

A multi-agent biomedical QA framework built upon Dempster-Shafer (D-S) evidence theory. The system integrates PubMed retrieval, structured evidence evaluation, D-S evidence fusion, and a dual-branch reasoning architecture combining symbolic evidence aggregation with direct LLM reasoning, followed by final arbitration.

Supports:
- Medical Multiple-choice QA (MedQA, MedMCQA)
- Medical Yes/No QA (PubMedQA)

---

## System Architecture

```
User Question
   │
   ▼
┌─────────────────────────────────────────────┐
│ Agent A: QuestionAnalyzer                   │
│ • EBM classification + PICO extraction      │
│ • Retrieval query generation                │
│ • Frame of Discernment (FoD) construction   │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Retriever: PubMed Retrieval                 │
│ • Hierarchical retrieval strategy           │
│ • Context-aware deduplication               │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Agent B: EvidenceAnalyzer                   │
│ • PICO extraction                           │
│ • Clinical summarization                    │
│ • Study design classification               │
└───────────────┬─────────────────────────────┘
                ▼
┌─────────────────────────────────────────────┐
│ Agent C: EvidenceEvaluator                  │
│ • Multi-label evidence evaluation           │
│ • BPA computation via rule engine           │
└───────────────┬─────────────────────────────┘
                ▼
┌─────────────────────────────────────────────┐
│ Agent D: EvidenceFusionEngine               │
│ • Dempster / Murphy fusion                  │
│ • Belief & plausibility estimation          │
│ • Final symbolic decision                   │
└───────────────┬─────────────────────────────┘
                ▼
┌─────────────────────────────────────────────┐
│ Agent E: ReportGenerator                    │
│ • Structured medical report generation      │
└───────────────┬─────────────────────────────┘
                ▼
┌─────────────────────────────────────────────┐
│ Final Arbiter                               │
│ • DS branch vs Direct LLM branch            │
│ • Conflict resolution & aggregation         │
└─────────────────────────────────────────────┘
```
---
## Project Structure

```
MEDAR-QA/
├── main.py
├── agents.py
├── llm_client.py
├── config.py
├── prompt.py
├── retriever.py
├── pubmed_online.py
├── pubmed.py
├── requirements.txt
│
├── medar_pipeline/
│   ├── pipeline.py
│   └── helpers.py
│
├── medar_agents/
│   ├── question_analyzer.py
│   ├── evidence_analyzer.py
│   ├── evidence_evaluator.py
│   ├── evidence_fusion_engine.py
│   ├── report_generator.py
│   ├── direct_reasoning_agent.py
│   ├── answer_arbiter.py
│   ├── completeness_controller.py
│   └── llm_chain.py
│
├── faiss_util/
├── datasets/
├── results/
└── TEST_RESULTS/
=======
## Quick Start

### 1) Install Dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure LLM Backend
Edit [`llm_client.py`](llm_client.py:13), set `LLM_BACKEND` and corresponding backend parameters:

- **Use Ollama** (recommended for local deployment):
  ```python
  LLM_BACKEND = "ollama"
  OLLAMA_URL = "http://localhost:11434"
  OLLAMA_MODEL = "qwen2.5:7b"
  ```

- **Use OpenAI Compatible API**:
  ```python
  LLM_BACKEND = "api"
  API_URL = "https://api.gptsapi.net/v1"
  API_KEY = "sk-..."
  MODEL_NAME = "gpt-4o-mini"
  ```

### 3) Run the System
=======

```bash
python main.py
In [`main.py`](main.py:8), replace `question` with your own medical question:

```python
question = """
A 20-year-old man comes to the physician because of worsening gait unsteadiness...
"A": "Renal cell carcinoma",
"B": "Meningioma",
"C": "Astrocytoma",
"D": "Vascular malformations"
"""
```

---

```
================================================================================
BELIEF Medical Evidence Reasoning System
================================================================================
Question: A 20-year-old man comes to the physician...
✔ Direct LLM Branch: Enabled (D-S branch and Direct LLM branch will run in parallel and be aggregated afterward)

[Step 1/6] QuestionAnalyzer: Question Analysis and Entity Extraction
→ Output: ebm_class, PICO, FoD = ["Renal cell carcinoma", "Meningioma", ...]

[Step 2/6] Retrieval: Evidence Retrieval Based on Extracted Entities
→ Hierarchical PubMed retrieval → Deduplication filtering

[Step 3/6] EvidenceAnalyzer: PICO Extraction and Study Type Classification
→ Structured analysis for each retrieved evidence

[Step 4/6] EvidenceEvaluator: Evidence Reliability Evaluation and BPA Computation
→ LLM-based multi-label classification → Rule-based BPA computation

[Step 5/6] EvidenceFusionEngine: Multi-Evidence Fusion and Decision Making
→ Dempster/Murphy fusion → Decision generation

[Step 6/6] ReportGenerator: Medical Evidence Report Generation
→ Final medical report

[Final Aggregation] Integrating D-S Branch and Direct LLM Branch Results
→ AnswerArbiter arbitration → Final answer + confidence score
```

---
## Retrieval Strategy

| Mode                    | Description                                  |
| ----------------------- | -------------------------------------------- |
| PubMed Online Retrieval | E-utilities API-based retrieval              |
| Local FAISS Retrieval   | Dense retrieval using prebuilt FAISS indexes |


The retrieval pipeline follows a hierarchical strategy:

1. Precise retrieval
2. Core concept retrieval
3. Broad recall
4. Original-query fallback
---

## Tech Stack

| Component    | Technology             |
| ------------ | ---------------------- |
| LLMs         | GPT-4o, DeepSeek, Qwen |
| Reasoning    | Dempster-Shafer Theory |
| Retrieval    | PubMed API + FAISS     |
| Framework    | LangChain LCEL         |
| NLP          | spaCy + SciSpaCy       |
| DL Framework | PyTorch + Transformers |
| Database     | Neo4j + MongoDB        |
| Inference Engine | vLLM |
---

## License

Released under the MIT License.

---

## 📧 Contact

For questions, suggestions, or potential collaborations, please reach out via:

- **GitHub Issues**: Submit an issue directly to our repository tracker.
- **Email**: ninghao@zust.edu.cn
- **Project Homepage**: ...

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a Star.⭐**

Made with ❤️ by BitLab Research Team

</div>
