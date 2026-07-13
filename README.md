# MEDAR-QA (BELIEF)

**Structured evidence modeling and uncertainty-aware fusion for biomedical question answering.**

MEDAR-QA is a research-oriented biomedical QA system that combines online PubMed retrieval, structured evidence analysis, Dempster-Shafer (D-S) evidence fusion, report generation, and optional arbitration with a direct LLM reasoning branch. It supports closed-set multiple-choice questions and biomedical yes/no questions.

The repository also provides separate MedRAG-PubMed and i-MedRAG-PubMed evaluation baselines. No benchmark result is claimed in this README; run the included evaluation scripts with the model and dataset configuration you intend to report.

> **Important:** This project is experimental software. It is not a medical device and must not be used as a substitute for professional diagnosis, treatment, or clinical judgment.

## Core architecture

The main MEDAR-QA workflow is a fixed, single-pass evidence pipeline:

```text
Question + optional user context
              |
              v
      QuestionAnalyzer
  (EBM/PICO structure, search terms, FoD)
              |
              v
       Online PubMed retrieval
  (staged keyword search with question fallback)
              |
              v
       EvidenceAnalyzer
  (evidence structure and clinical summary)
              |
        +-----+------------------+
        |                        |
        v                        v
 EvidenceEvaluator      DirectReasoningAgent
 (quality labels and       (optional branch)
  BPA construction)              |
        |                        |
        v                        |
 EvidenceFusionEngine            |
 (D-S/Murphy fusion)             |
        |                        |
        v                        |
   ReportGenerator               |
        |                        |
        +-----------+------------+
                    v
             AnswerArbiter
        (when the direct branch is enabled)
                    |
                    v
          Timestamped JSON result
```

The core workflow performs one retrieval and evidence-evaluation pass. Iterative retrieval is intentionally limited to the independent i-MedRAG-PubMed baseline described below.

The active retriever in the main pipeline uses NCBI PubMed E-utilities. The repository contains local FAISS utilities for separate indexing and retrieval experiments, but they are not part of the default `run_pipeline` path.

## Repository layout

```text
MEDAR-QA/
|-- main.py                    # Runnable core-pipeline example
|-- agents.py                  # Public/compatibility exports for agents
|-- llm_client.py              # Unified API, Ollama, vLLM, and Transformers clients
|-- prompt.py                  # Agent prompt templates
|-- retriever.py               # Main staged PubMed retrieval strategy
|-- pubmed_online.py           # NCBI E-utilities client
|-- medar_agents/              # Analysis, evaluation, fusion, reporting, arbitration
|-- medar_pipeline/            # Single-pass orchestration and helper functions
|-- medrag/                    # MedRAG and i-MedRAG baseline implementations
|-- run_experiment.py          # Unified baseline evaluation entry point
|-- test_script/               # Maintained baseline and benchmarking entry points
|-- test_script/tools/         # Cost benchmarking and result analysis
|-- tests/                     # Offline unit and regression tests
|-- datasets/                  # Repository evaluation datasets
|-- config/                    # Environment-based settings
|-- faiss_util/                # Optional local FAISS utilities
|-- .env.example               # Configuration template
`-- requirements.txt           # Python dependencies
```

## Installation

Python 3.10 or newer is required by the current type syntax.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some configurations download model weights or embedding models on first use. The online retrieval paths require access to NCBI PubMed.

## Configuration

Copy the template before running the system:

```bash
# Linux or macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Configuration is loaded from the repository-root `.env` file by [`config/settings.py`](config/settings.py). Do not commit `.env`; it is excluded by `.gitignore`.

### LLM backends

Set `MEDAR_LLM_BACKEND` to one of the supported values and configure the matching variables:

| Backend | Required settings |
| --- | --- |
| `ollama` | `MEDAR_OLLAMA_URL`, `MEDAR_OLLAMA_MODEL` |
| `api` | `MEDAR_API_URL`, `MEDAR_API_KEY`, `MEDAR_MODEL_NAME` |
| `vllm` | `MEDAR_VLLM_URL`, `MEDAR_VLLM_MODEL` |
| `transformers` | `MEDAR_LOCAL_MODEL_PATH`, `MEDAR_DEVICE` |

`MEDAR_DISABLE_THINKING` controls supported model-specific thinking output. Its default value is `true`; model overrides are defined in `config/settings.py`.

### PubMed and optional services

| Variable | Purpose |
| --- | --- |
| `MEDAR_NCBI_TOOL` | Tool identifier sent to NCBI |
| `MEDAR_NCBI_EMAIL` | Contact email sent to NCBI |
| `MEDAR_NCBI_API_KEY` | Optional NCBI API key |
| `MEDAR_NEO4J_*`, `MEDAR_MONGO_*` | Optional database settings used by auxiliary code |
| `MEDAR_FAISS_*`, `MEDAR_EMBEDDING_MODEL_*` | Optional local-index and embedding settings |

Use a real contact email for sustained PubMed evaluation runs. Keep all API keys and credentials in `.env`, never in source code or result files.

## Quick start

The simplest smoke run uses the example question in [`main.py`](main.py):

```bash
python main.py
```

For programmatic use, call the pipeline directly:

```python
from medar_pipeline import run_pipeline

result = run_pipeline(
    question="""
    Which condition is associated with a pathogenic NF2 mutation?
    "A": "Renal cell carcinoma"
    "B": "Meningioma"
    "C": "Astrocytoma"
    "D": "Vascular malformation"
    """,
    context="",
    task_mode="SELECTION",
    enable_direct_llm_branch=True,
    output_dir="TEST_RESULTS",
)
```

`run_pipeline` creates the output directory when necessary and writes a timestamped JSON record. Set `enable_direct_llm_branch=False` to run only the structured D-S branch and report generator.

Compatibility note: the former `max_rounds` argument, completeness controller, and round-history fields were removed when the core workflow became single-pass. Existing callers should remove `max_rounds`; use the returned `retrieval_summary` for retrieval counts. `output_dir` is keyword-only to prevent an old positional round count from being misinterpreted as a path.

### Task modes

| Mode | Use case | Expected closed-set answer |
| --- | --- | --- |
| `SELECTION` | Multiple-choice medical QA; include labeled options in the question text | One option label, such as `A`, `B`, `C`, or `D` |
| `YES_NO` | Biomedical yes/no QA | `yes`, `no`, or `maybe` |

Example yes/no call:

```python
result = run_pipeline(
    question="Does the retrieved clinical evidence support the stated association?",
    task_mode="YES_NO",
)
```

The uppercase names above are the canonical API values; common case variants and `YES-NO`, `YES/NO`, or `YESNO` aliases are normalized. The selected mode controls the direct-reasoning, report-generation, and final-arbitration prompt contracts.

## Retrieval behavior

For the core pipeline, `QuestionAnalyzer` supplies prioritized biomedical keywords. `retriever.py` then tries progressively broader Title/Abstract PubMed queries and stops at the first stage that returns papers. If those queries return nothing, the original question is cleaned and used as a fallback query. Retrieved records are converted to title/abstract evidence objects before evidence analysis.

User-provided `context` is added as evidence. Retrieved items that strongly overlap with that context are filtered before evaluation.

## Evaluation baselines

The baseline implementations are independent of the core MEDAR-QA single-pass pipeline. The unified entry point supports `MedRAG-PubMed` and `i-MedRAG-PubMed` on PubMedQA, MedQA, and MedMCQA.

### MedRAG-PubMed

The MedRAG baseline follows this path:

```text
question only
  -> online PubMed candidate retrieval
  -> local Okapi BM25 ranking over title + abstract
  -> top-k evidence
  -> closed-set LLM JSON answer
  -> dataset-specific normalization and evaluation
```

Only the question enters retrieval. Candidate options and gold labels are not accepted by the retriever, and PubMedQA's dataset-provided `CONTEXTS` are not used as retrieval evidence. Raw PubMed candidates are cached under `cache/medrag_pubmed` by default.

Run small evaluations:

```bash
python run_experiment.py --method MedRAG-PubMed --dataset pubmedqa --limit 5
python run_experiment.py --method MedRAG-PubMed --dataset medqa --limit 20 --top-k 5 --candidate-k 50
python run_experiment.py --method MedRAG-PubMed --dataset medmcqa --limit 20 --top-k 5 --candidate-k 50
```

Results are written under `TEST_RESULTS/medrag/<dataset>/`. Useful controls include `--top-k`, `--candidate-k`, `--temperature`, `--cache-dir`, `--refresh-cache`, `--omit-prompt`, and `--omit-context`.

Each result records the question-only retrieval query, ranked PMID/title/abstract snippets with BM25 scores, the normalized prediction, parse status, raw model output, and timing/token fields unless the corresponding omission flag is enabled. The file-level summary reports strict accuracy, Macro-F1, invalid outputs, retrieval counts, and per-label metrics.

### i-MedRAG-PubMed

i-MedRAG-PubMed adds iterative follow-up retrieval to the same online PubMed/BM25 foundation:

```text
question + candidate answers
  -> generate follow-up PubMed queries
  -> retrieve and answer each follow-up query
  -> accumulate query-answer history
  -> select one final closed-set answer
```

Run one dataset:

```bash
python run_experiment.py --method i-MedRAG-PubMed --dataset MedQA \
  --backbone your-model-id \
  --n-rounds 2 --n-queries 2 --k-per-query 5 \
  --output TEST_RESULTS/imedrag_pubmed/medqa/run.jsonl
```

Run all three supported datasets and print the accuracy/Macro-F1 table:

```bash
python run_experiment.py --method i-MedRAG-PubMed --dataset all \
  --n-rounds 2 --n-queries 2 --k-per-query 5
```

Default results are written under `TEST_RESULTS/imedrag_pubmed/<dataset>/`. Retrieval and LLM responses are cached separately unless `--no-cache` or `--no-llm-cache` is supplied. Logs include follow-up rounds, queries, retrieved PMID/title data, normalized predictions, parsing status, token estimates, PubMed query/document counts, latency, accuracy, and Macro-F1.

## Testing and cost analysis

Run all discoverable unit tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The MedRAG and i-MedRAG unit tests use fake PubMed and LLM clients, so these focused tests do not require network access or model weights:

```bash
python -m unittest tests.test_medrag tests.test_imedrag_pubmed -v
```

For a small online PubMedQA cost/latency comparison of the two PubMed baselines:

```bash
python test_script/tools/benchmark_pubmedqa_costs.py \
  --data-path datasets/pubmedqa.json \
  --methods medrag,imedrag \
  --limit 5
```

This benchmark calls the configured LLM and may call PubMed; it is not an offline unit test. To summarize already-generated MedRAG/i-MedRAG logs with the same cost fields, run:

```bash
python test_script/tools/analyze_medrag_costs.py --dataset pubmedqa
```

When a backend does not return usage metadata, token counts are estimates rather than tokenizer-exact measurements.

## Outputs, caching, and privacy

- Core runs and evaluations can store the original question, optional context, retrieved abstracts and identifiers, prompts, raw model responses, intermediate agent outputs, predictions, and metrics.
- `TEST_RESULTS/`, `cache/`, and `.env` are ignored by Git by default. Review staged files before every commit rather than relying only on ignore rules.
- Questions and evidence may be sent to NCBI and to the configured remote LLM service. Use local backends where appropriate and do not process identifiable patient data without an approved privacy and governance process.
- Use `--omit-prompt`, `--omit-context`, or `--omit-retrieval` where supported when retaining full intermediate text is unnecessary.

## Limitations

- Retrieval quality depends on question analysis, PubMed availability, and abstract-level evidence; full-text evidence is not retrieved by the default online path.
- LLM-generated evidence labels, reports, and arbitration decisions can be incorrect or inconsistent across models.
- D-S confidence and BPA values are reasoning artifacts of this implementation, not calibrated clinical probabilities.
- Baseline token counts can be approximate, and wall-clock comparisons depend on backend, cache state, network conditions, and hardware.
- Included datasets and scripts are research/evaluation assets; users remain responsible for dataset licensing, reporting methodology, and leakage checks for their experiments.

## License

This project is released under the [MIT License](LICENSE).
