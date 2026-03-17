"""
SELF-RAG Baseline Implementation
=================================
基于论文: "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
(Asai et al., 2023) https://arxiv.org/abs/2310.11511

本脚本为 **推理阶段模拟版** Self-RAG：
  - 不需要特殊训练的模型权重
  - 通过提示词工程让普通指令微调 LLM 输出反射标签 (Reflection Tokens)
  - 作为 Baseline 用于对比评估 PubMedQA / MedQA 数据集上的准确率

反射标签体系 (Reflection Token Schema)
--------------------------------------
[Retrieve=Yes]  / [Retrieve=No]
[ISREL=Relevant] / [ISREL=Irrelevant]
[ISSUP=Fully supported] / [ISSUP=Partially supported] / [ISSUP=No support]
[ISUSE=1] ~ [ISUSE=5]

评分公式 (Heuristic Scoring Formula)
------------------------------------
S(candidate) = w_rel * s_rel + w_sup * s_sup + w_use * s_use

  ┌──────────────┬───────────────────────────────────────┐
  │  Tag         │  Numeric Value                        │
  ├──────────────┼───────────────────────────────────────┤
  │  ISREL       │  Relevant=1.0   Irrelevant=0.0        │
  │  ISSUP       │  Fully=1.0   Partially=0.5   No=0.0   │
  │  ISUSE       │  (raw score / 5.0)  ∈ [0.2, 1.0]     │
  └──────────────┴───────────────────────────────────────┘

# ======= 用户自定义区域（可在此修改评分权重与数据路径）=======
W_REL  = 0.3   # ISREL 权重
W_SUP  = 0.4   # ISSUP 权重
W_USE  = 0.3   # ISUSE 权重
# ============================================================
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from tqdm import tqdm

# ============================================================
#  复用项目现有 LLM 调用接口
# ============================================================
from llm_client import call_llm
from pubmed_online import PubMedOnlineSearcher


# ============================================================
#  全局配置
# ============================================================

# ---------- 评分权重 ----------
# 可根据下游任务调整; 三者之和应为 1.0（非强制）
W_REL: float = 0.3   # ISREL 相关性权重
W_SUP: float = 0.4   # ISSUP 支持度权重
W_USE: float = 0.3   # ISUSE 有用性权重

# ---------- 检索参数 ----------
TOP_K: int = 3       # 并行检索的文档数量（论文中的 K）

# ---------- 生成参数 ----------
LLM_TEMPERATURE: float = 0.0   # 推理阶段使用贪心解码 (temperature=0)
LLM_MAX_TOKENS:  int   = 4096  # 每次生成的最大 token 数（需为思考模型预留空间）

# ============================================================
#  系统提示词 (System Prompts)
# ============================================================

# ------ 步骤 1: 检索判断 ------
PROMPT_RETRIEVE_DECISION = """\
You are a biomedical question-answering assistant that decides whether external document retrieval is needed.

Given the following question, decide if you need to retrieve external documents to answer it accurately.

Question:
{question}

Instructions:
- If the question requires specific clinical evidence, recent findings, or factual details you are uncertain about, output [Retrieve=Yes].
- If you can confidently answer based on your internal knowledge (e.g., simple definitions, well-known facts), output [Retrieve=No].
- After the tag, provide a brief justification (1-2 sentences).

Your response MUST start with either [Retrieve=Yes] or [Retrieve=No].
"""

# ------ 无检索直接生成 ------
PROMPT_GENERATE_NO_RETRIEVAL = """\
You are a biomedical expert. Answer the following question based solely on your internal knowledge.

Question:
{question}

Instructions:
- Provide a clear, concise answer.
- At the end of your answer, add an [ISUSE=N] tag (N from 1 to 5) rating how useful/confident your answer is.
  1 = very uncertain, 5 = very confident and highly useful.

Format:
<your answer here>
[ISUSE=N]
"""

# ------ 步骤 2: 带文档的生成（评估相关性 + 生成回答 + 评估支撑度 + 有用性）------
PROMPT_GENERATE_WITH_DOC = """\
You are a biomedical expert performing a Retrieval-Augmented Generation task.

Question:
{question}

Retrieved Document:
\"\"\"
{document}
\"\"\"

Instructions:
Step 1 - Relevance: First, evaluate whether the retrieved document is relevant to the question.
  Output: [ISREL=Relevant] or [ISREL=Irrelevant]

Step 2 - Answer: Generate a concise answer segment to the question, using evidence from the document.

Step 3 - Support: Evaluate whether the document fully supports your answer.
  Output: [ISSUP=Fully supported] or [ISSUP=Partially supported] or [ISSUP=No support]

Step 4 - Usefulness: Rate the overall usefulness of this answer on a scale of 1-5.
  Output: [ISUSE=N]  (1=not useful, 5=extremely useful)

Your response MUST follow EXACTLY this format:
[ISREL=...] <your answer segment here> [ISSUP=...] [ISUSE=N]

Do NOT add extra text before [ISREL=...].
"""

# ------ PubMedQA 专用: 问题需 Yes/No/Maybe ------
PROMPT_PUBMEDQA_NO_RETRIEVAL = """\
You are a biomedical expert. Answer the following yes/no/maybe question based on your internal knowledge.

Question:
{question}

Instructions:
- Your final answer MUST be one of: yes, no, maybe
- Provide a brief reasoning before your final answer.
- At the end, add [ISUSE=N] rating your confidence (1-5).

Format:
Reasoning: <1-2 sentence reasoning>
Answer: <yes|no|maybe>
[ISUSE=N]
"""

PROMPT_PUBMEDQA_WITH_DOC = """\
You are a biomedical expert performing a Retrieval-Augmented Generation task.

Question (Yes/No/Maybe):
{question}

Retrieved Document:
\"\"\"
{document}
\"\"\"

Instructions:
Step 1 - Relevance: Evaluate if the document is relevant.
  Output: [ISREL=Relevant] or [ISREL=Irrelevant]

Step 2 - Answer: Provide reasoning and your yes/no/maybe answer.
  Format:
  Reasoning: <brief reasoning>
  Answer: <yes|no|maybe>

Step 3 - Support: Does the document support your answer?
  Output: [ISSUP=Fully supported] or [ISSUP=Partially supported] or [ISSUP=No support]

Step 4 - Usefulness: Rate answer usefulness 1-5.
  Output: [ISUSE=N]

Your response MUST follow this exact format:
[ISREL=...] Reasoning: <reasoning> Answer: <yes|no|maybe> [ISSUP=...] [ISUSE=N]
"""

# ------ MedQA 专用: MCQ A/B/C/D ------
PROMPT_MEDQA_NO_RETRIEVAL = """\
You are a medical expert. Answer the following multiple-choice question based on your internal knowledge.

Question:
{question}

Options:
{options}

Instructions:
- Choose the single best answer (A, B, C, or D).
- Provide brief reasoning.
- At the end, add [ISUSE=N] rating your confidence (1-5).

Format:
Reasoning: <brief reasoning>
Answer: <A|B|C|D>
[ISUSE=N]
"""

PROMPT_MEDQA_WITH_DOC = """\
You are a medical expert performing a Retrieval-Augmented Generation task.

Question:
{question}

Options:
{options}

Retrieved Document:
\"\"\"
{document}
\"\"\"

Instructions:
Step 1 - Relevance: Is this document relevant to answering the question?
  Output: [ISREL=Relevant] or [ISREL=Irrelevant]

Step 2 - Answer: Choose the best answer (A/B/C/D) with brief reasoning.
  Format:
  Reasoning: <brief reasoning>
  Answer: <A|B|C|D>

Step 3 - Support: Does the document support your answer?
  Output: [ISSUP=Fully supported] or [ISSUP=Partially supported] or [ISSUP=No support]

Step 4 - Usefulness: Rate answer usefulness 1-5.
  Output: [ISUSE=N]

Your response MUST follow this exact format:
[ISREL=...] Reasoning: <reasoning> Answer: <A|B|C|D> [ISSUP=...] [ISUSE=N]
"""


# ============================================================
#  反射标签解析器 (Reflection Token Parser)
# ============================================================

class ReflectionTokenParser:
    """
    从 LLM 生成文本中解析 SELF-RAG 反射标签。
    对格式的容忍度较高（不区分大小写，允许空格变体）。
    """

    # ---- 正则模式 ----
    _RE_RETRIEVE = re.compile(r'\[Retrieve=(Yes|No)\]', re.IGNORECASE)
    _RE_ISREL    = re.compile(r'\[ISREL=(Relevant|Irrelevant)\]', re.IGNORECASE)
    _RE_ISSUP    = re.compile(
        r'\[ISSUP=(Fully\s+supported|Partially\s+supported|No\s+support)\]',
        re.IGNORECASE
    )
    _RE_ISUSE    = re.compile(r'\[ISUSE=([1-5])\]', re.IGNORECASE)
    _RE_ANSWER_YESNO  = re.compile(r'Answer:\s*(yes|no|maybe)', re.IGNORECASE)
    _RE_ANSWER_MCQ    = re.compile(r'Answer:\s*([A-D])', re.IGNORECASE)

    @classmethod
    def parse_retrieve(cls, text: str) -> Optional[str]:
        """返回 'Yes' / 'No' / None"""
        m = cls._RE_RETRIEVE.search(text)
        return m.group(1).capitalize() if m else None

    @classmethod
    def parse_isrel(cls, text: str) -> Optional[str]:
        """返回 'Relevant' / 'Irrelevant' / None"""
        m = cls._RE_ISREL.search(text)
        return m.group(1).capitalize() if m else None

    @classmethod
    def parse_issup(cls, text: str) -> Optional[str]:
        """返回标准化字符串 / None"""
        m = cls._RE_ISSUP.search(text)
        if not m:
            return None
        raw = m.group(1).lower()
        if 'fully' in raw:
            return 'Fully supported'
        if 'partially' in raw:
            return 'Partially supported'
        return 'No support'

    @classmethod
    def parse_isuse(cls, text: str) -> Optional[int]:
        """返回 1-5 整数 / None"""
        m = cls._RE_ISUSE.search(text)
        return int(m.group(1)) if m else None

    @classmethod
    def parse_answer_yesno(cls, text: str) -> str:
        """从文本提取 yes/no/maybe 答案"""
        m = cls._RE_ANSWER_YESNO.search(text)
        if m:
            return m.group(1).lower()
        # 降级：直接扫描
        text_lower = text.lower()
        for kw in ('yes', 'no', 'maybe'):
            if kw in text_lower:
                return kw
        return ''

    @classmethod
    def parse_answer_mcq(cls, text: str) -> str:
        """从文本提取 A/B/C/D 答案"""
        m = cls._RE_ANSWER_MCQ.search(text)
        if m:
            return m.group(1).upper()
        # 降级：最后出现的单独字母
        for ch in reversed(text.upper()):
            if ch in 'ABCD':
                return ch
        return ''

    @classmethod
    def extract_answer_segment(cls, text: str) -> str:
        """提取 [ISREL=...] 与 [ISSUP=...] 之间的内容作为回答片段"""
        # 先去掉开头的 [ISREL=...] 标签
        stripped = cls._RE_ISREL.sub('', text).strip()
        # 再截断 [ISSUP=...] 之后的内容
        issup_pos = cls._RE_ISSUP.search(stripped)
        if issup_pos:
            return stripped[:issup_pos.start()].strip()
        return stripped


# ============================================================
#  候选项评分器 (Candidate Scorer)
# ============================================================

class CandidateScorer:
    """
    根据 SELF-RAG 论文中的启发式评分公式计算每个候选项得分：

      S(candidate) = w_rel * s_rel + w_sup * s_sup + w_use * s_use

    由于无法获得原论文中 token 概率，使用离散数值代替。

    # 在此处注入自定义权重
    # W_REL, W_SUP, W_USE 为全局变量，可在脚本顶部修改
    """

    # ---- 标签到数值的映射 ----
    ISREL_SCORES: Dict[str, float] = {
        'relevant':   1.0,
        'irrelevant': 0.0,
    }
    ISSUP_SCORES: Dict[str, float] = {
        'fully supported':    1.0,
        'partially supported': 0.5,
        'no support':         0.0,
    }

    @classmethod
    def isrel_to_score(cls, tag: Optional[str]) -> float:
        if tag is None:
            return 0.5   # 缺失时给中性分
        return cls.ISREL_SCORES.get(tag.lower(), 0.5)

    @classmethod
    def issup_to_score(cls, tag: Optional[str]) -> float:
        if tag is None:
            return 0.5
        return cls.ISSUP_SCORES.get(tag.lower(), 0.5)

    @classmethod
    def isuse_to_score(cls, tag: Optional[int]) -> float:
        if tag is None:
            return 0.5
        return tag / 5.0   # 归一化到 [0.2, 1.0]

    @classmethod
    def compute(cls, isrel: Optional[str], issup: Optional[str],
                isuse: Optional[int]) -> float:
        """
        计算综合得分。
        使用全局权重 W_REL, W_SUP, W_USE。
        """
        s_rel = cls.isrel_to_score(isrel)
        s_sup = cls.issup_to_score(issup)
        s_use = cls.isuse_to_score(isuse)
        return W_REL * s_rel + W_SUP * s_sup + W_USE * s_use


# ============================================================
#  检索器 (Retriever)
#  直接复用项目的 PubMedOnlineSearcher
# ============================================================

class SelfRAGRetriever:
    """
    轻量级 PubMed 检索器封装，专为 SELF-RAG 推理流程设计。
    若需替换为本地 FAISS 检索，在 retrieve() 方法中替换即可。
    """

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self._searcher = PubMedOnlineSearcher(top_k=top_k)

    def retrieve(self, query: str) -> List[str]:
        """
        给定查询，返回 top_k 个文档字符串列表。
        每个文档格式: "Title: ...\nAbstract: ..."

        可在此处替换为任意检索后端：
          - FAISS 本地检索: from faiss_util.bio_faiss import ...
          - 现有 retrieve_process(): from retriever import retrieve_process
        """
        try:
            papers = self._searcher.search(query)
        except Exception as e:
            print(f"[Retriever] PubMed 检索失败: {e}，返回空列表")
            return []

        docs = []
        for paper in papers:
            title    = paper.get('title', 'Unknown Title')
            abstract = paper.get('abstract', '')
            if len(abstract) > 1200:
                abstract = abstract[:1200].rsplit(' ', 1)[0] + '...'
            docs.append(f"Title: {title}\nAbstract: {abstract}")

        return docs[:self.top_k]


# ============================================================
#  SELF-RAG 核心推理流程
# ============================================================

class SelfRAGPipeline:
    """
    SELF-RAG 推理管线。

    推理算法（三步）:
    ─────────────────
    Step 1 | On-Demand Retrieval
        → 询问 LLM 是否需要检索
        → [Retrieve=No]: 直接生成答案 + [ISUSE]
        → [Retrieve=Yes]: 进入 Step 2

    Step 2 | Parallel Generation with K Documents
        → 对 K 个检索文档，各独立生成候选回答
        → 每个候选包含: [ISREL] → 回答片段 → [ISSUP] → [ISUSE]

    Step 3 | Segment Scoring & Selection
        → 用 S = w_rel*s_rel + w_sup*s_sup + w_use*s_use 对每个候选评分
        → 返回分数最高的候选

    Args:
        task_type: 'pubmedqa' | 'medqa' | 'general'
        retriever: SelfRAGRetriever 实例（可替换）
    """

    def __init__(
        self,
        task_type:  str = 'general',
        retriever:  Optional[SelfRAGRetriever] = None,
        top_k:      int = TOP_K,
    ):
        self.task_type = task_type
        self.retriever = retriever or SelfRAGRetriever(top_k=top_k)
        self.parser    = ReflectionTokenParser()
        self.scorer    = CandidateScorer()

    # ----------------------------------------------------------
    # Step 1: 检索决策
    # ----------------------------------------------------------

    def decide_retrieve(self, question: str, options_str: str = '') -> Tuple[str, str]:
        """
        询问 LLM 是否需要检索。

        Returns:
            (decision, raw_response)
            decision: 'Yes' | 'No'
        """
        full_q = question
        if options_str:
            full_q = f"{question}\n\nOptions:\n{options_str}"

        prompt   = PROMPT_RETRIEVE_DECISION.format(question=full_q)
        response = call_llm(prompt, temperature=LLM_TEMPERATURE, max_tokens=1024)
        decision = self.parser.parse_retrieve(response)

        if decision is None:
            # 默认策略: 对生医问题通常需要检索
            decision = 'Yes'

        return decision, response

    # ----------------------------------------------------------
    # Step 2a: 无检索直接生成
    # ----------------------------------------------------------

    def generate_no_retrieval(self, question: str,
                              options_str: str = '') -> Dict[str, Any]:
        """
        不使用检索文档，直接生成答案。
        返回标准化候选字典。
        """
        if self.task_type == 'pubmedqa':
            prompt = PROMPT_PUBMEDQA_NO_RETRIEVAL.format(question=question)
        elif self.task_type == 'medqa':
            prompt = PROMPT_MEDQA_NO_RETRIEVAL.format(
                question=question, options=options_str
            )
        else:
            prompt = PROMPT_GENERATE_NO_RETRIEVAL.format(question=question)

        response    = call_llm(prompt, temperature=LLM_TEMPERATURE,
                               max_tokens=LLM_MAX_TOKENS)
        isuse_tag   = self.parser.parse_isuse(response)
        score       = self.scorer.compute(isrel='Relevant', issup='Fully supported',
                                          isuse=isuse_tag)

        return {
            'doc':     None,
            'raw':     response,
            'segment': response,
            'isrel':   'Relevant',       # 无文档，视为自身相关
            'issup':   'Fully supported', # 内部知识假设支撑
            'isuse':   isuse_tag,
            'score':   score,
            'answer':  self._extract_answer(response),
        }

    # ----------------------------------------------------------
    # Step 2b: 带文档生成（并行 K 个候选）
    # ----------------------------------------------------------

    def generate_with_doc(self, question: str, doc: str,
                          options_str: str = '') -> Dict[str, Any]:
        """
        给定单个检索文档，生成带反射标签的候选回答。
        """
        if self.task_type == 'pubmedqa':
            prompt = PROMPT_PUBMEDQA_WITH_DOC.format(
                question=question, document=doc
            )
        elif self.task_type == 'medqa':
            prompt = PROMPT_MEDQA_WITH_DOC.format(
                question=question, options=options_str, document=doc
            )
        else:
            prompt = PROMPT_GENERATE_WITH_DOC.format(
                question=question, document=doc
            )

        response = call_llm(prompt, temperature=LLM_TEMPERATURE,
                            max_tokens=LLM_MAX_TOKENS)

        isrel   = self.parser.parse_isrel(response)
        issup   = self.parser.parse_issup(response)
        isuse   = self.parser.parse_isuse(response)
        segment = self.parser.extract_answer_segment(response)
        score   = self.scorer.compute(isrel, issup, isuse)
        answer  = self._extract_answer(response)

        return {
            'doc':     doc[:200] + '...' if len(doc) > 200 else doc,
            'raw':     response,
            'segment': segment,
            'isrel':   isrel,
            'issup':   issup,
            'isuse':   isuse,
            'score':   score,
            'answer':  answer,
        }

    # ----------------------------------------------------------
    # Step 3: 候选选择
    # ----------------------------------------------------------

    def select_best_candidate(self, candidates: List[Dict]) -> Dict:
        """
        根据综合得分选择最优候选项。
        """
        return max(candidates, key=lambda c: c['score'])

    # ----------------------------------------------------------
    #  主推理入口
    # ----------------------------------------------------------

    def run(self, question: str,
            options: Optional[Dict[str, str]] = None,
            provided_docs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        完整 SELF-RAG 推理流程。

        Args:
            question:      问题文本
            options:       MCQ 选项字典（MedQA 使用），如 {'A': '...', 'B': '...', ...}
            provided_docs: 预先提供的文档列表（如 PubMedQA 自带的 CONTEXTS）。
                           若不为 None，直接跳过在线检索，使用这些文档作为候选。
                           这对应标准的 "oracle / with-context" 评估模式。

        Returns:
            {
                'answer':     最终答案字符串
                'best':       最优候选详情字典
                'candidates': 全部候选列表
                'retrieved':  是否执行了检索
                'docs':       使用的文档列表
                'retrieve_response': 检索决策的原始响应
            }
        """
        options_str = self._format_options(options) if options else ''

        # ── Step 1: 检索决策 ──────────────────────────────────────
        # 若已注入 provided_docs，直接跳过检索决策步骤，强制走 Retrieve=Yes 路径
        if provided_docs is not None:
            retrieve_decision = 'Yes (provided)'
            retrieve_response = '[Retrieve=Yes] (bypassed: dataset CONTEXTS injected)'
            print(f"  [Step1] Retrieve Decision: {retrieve_decision}")
            docs = provided_docs[:TOP_K]
            print(f"  [Step2] Using {len(docs)} provided CONTEXTS docs (no online retrieval)")
        else:
            retrieve_decision, retrieve_response = self.decide_retrieve(
                question, options_str
            )
            print(f"  [Step1] Retrieve Decision: {retrieve_decision}")

            if retrieve_decision == 'No':
                # 无检索直接生成
                candidate = self.generate_no_retrieval(question, options_str)
                return {
                    'answer':           candidate['answer'],
                    'best':             candidate,
                    'candidates':       [candidate],
                    'retrieved':        False,
                    'docs':             [],
                    'retrieve_response': retrieve_response,
                }

            # ── Step 2: 在线检索 + 并行生成 K 个候选 ────────────────
            docs = self.retriever.retrieve(question)
            print(f"  [Step2] Retrieved {len(docs)} docs")

        if not docs:
            # 检索失败，降级到无检索模式
            print("  [Step2] Fallback to no-retrieval generation")
            candidate = self.generate_no_retrieval(question, options_str)
            return {
                'answer':           candidate['answer'],
                'best':             candidate,
                'candidates':       [candidate],
                'retrieved':        False,
                'docs':             [],
                'retrieve_response': retrieve_response,
            }

        candidates = []
        for i, doc in enumerate(docs):
            print(f"  [Step2] Generating candidate {i+1}/{len(docs)} ...")
            cand = self.generate_with_doc(question, doc, options_str)
            candidates.append(cand)
            print(f"          ISREL={cand['isrel']}, ISSUP={cand['issup']}, "
                  f"ISUSE={cand['isuse']}, Score={cand['score']:.3f}")

        # ── Step 3: 评分 & 选择 ───────────────────────────────────
        best = self.select_best_candidate(candidates)
        print(f"  [Step3] Best score={best['score']:.3f}, answer='{best['answer']}'")

        return {
            'answer':           best['answer'],
            'best':             best,
            'candidates':       candidates,
            'retrieved':        True,
            'docs':             docs,
            'retrieve_response': retrieve_response,
        }

    # ----------------------------------------------------------
    #  内部工具
    # ----------------------------------------------------------

    def _extract_answer(self, text: str) -> str:
        """根据任务类型提取最终答案"""
        if self.task_type == 'pubmedqa':
            return self.parser.parse_answer_yesno(text)
        elif self.task_type == 'medqa':
            return self.parser.parse_answer_mcq(text)
        else:
            return self.parser.extract_answer_segment(text)

    @staticmethod
    def _format_options(options: Dict[str, str]) -> str:
        return '\n'.join(f'{k}: {v}' for k, v in options.items())


# ============================================================
#  PubMedQA 评估器
# ============================================================

# ---- 数据集路径（可修改）----
PUBMEDQA_DATA_PATH = "data/pubmedqa_sample.json"
PUBMEDQA_OUTPUT_DIR = "TEST_RESULTS/self_rag/pubmedqa"

# ---- 测试数量（None = 全量）----
PUBMEDQA_TEST_LIMIT = None

# ---- 是否直接使用数据集自带的 CONTEXTS（黄金段落）作为检索文档 ----
# True  → oracle / with-context 模式：直接注入题目附带的 CONTEXTS，不做在线检索
#          与 test_pubmedqa_with_context.py 的评估设置一致，通常准确率更高
# False → 纯检索模式：通过 PubMed 在线检索获取文档（端到端评估）
PUBMEDQA_USE_PROVIDED_CONTEXT = True

# ---- 保存间隔 ----
PUBMEDQA_SAVE_INTERVAL = 50


class PubMedQASelfRAGEvaluator:
    """
    在 PubMedQA 数据集上评估 Self-RAG Baseline。
    数据集格式: {pmid: {QUESTION, CONTEXTS, final_decision (yes/no/maybe)}}
    """

    def __init__(self):
        self.pipeline    = SelfRAGPipeline(task_type='pubmedqa')
        self.results: List[Dict] = []
        self.correct_count = 0
        self.total_count   = 0
        os.makedirs(PUBMEDQA_OUTPUT_DIR, exist_ok=True)

    def load_data(self) -> List[Dict]:
        with open(PUBMEDQA_DATA_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        data = []
        for pmid, item in raw.items():
            item['pmid'] = pmid
            data.append(item)
            if PUBMEDQA_TEST_LIMIT and len(data) >= PUBMEDQA_TEST_LIMIT:
                break
        return data

    @staticmethod
    def normalize_answer(ans: str) -> str:
        ans = (ans or '').lower().strip()
        if ans in ('yes', 'positive'):
            return 'yes'
        if ans in ('no', 'negative'):
            return 'no'
        if ans in ('maybe', 'uncertain', 'insufficient_evidence'):
            return 'maybe'
        return ans

    def run_single(self, item: Dict) -> Dict:
        pmid         = item.get('pmid', 'unknown')
        question     = item.get('QUESTION', '')
        ground_truth = self.normalize_answer(item.get('final_decision', ''))

        # 构建 provided_docs：将 CONTEXTS 列表直接转为文档字符串
        # PubMedQA 的 CONTEXTS 是与该题目对应的黄金摘要段落列表
        provided_docs = None
        if PUBMEDQA_USE_PROVIDED_CONTEXT:
            contexts = item.get('CONTEXTS', [])
            if contexts:
                provided_docs = [f"Abstract {i+1}: {ctx}" for i, ctx in enumerate(contexts)]

        try:
            result    = self.pipeline.run(question, provided_docs=provided_docs)
            predicted = self.normalize_answer(result['answer'])
            correct   = (predicted == ground_truth)

            return {
                'pmid':       pmid,
                'question':   question,
                'gt':         ground_truth,
                'pred':       predicted,
                'correct':    correct,
                'retrieved':  result['retrieved'],
                'best_score': result['best']['score'],
                'best_isrel': result['best']['isrel'],
                'best_issup': result['best']['issup'],
                'best_isuse': result['best']['isuse'],
                'raw_answer': result['best']['raw'],
                'error':      None,
            }
        except Exception as e:
            import traceback
            return {
                'pmid': pmid, 'question': question,
                'gt': ground_truth, 'pred': '', 'correct': False,
                'retrieved': False, 'best_score': 0.0,
                'best_isrel': None, 'best_issup': None, 'best_isuse': None,
                'raw_answer': '', 'error': str(e),
            }

    def save_results(self, filename: Optional[str] = None):
        if filename is None:
            ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(PUBMEDQA_OUTPUT_DIR,
                                    f'self_rag_pubmedqa_{ts}.json')
        accuracy = self.correct_count / self.total_count if self.total_count else 0
        output   = {
            'meta': {
                'total':    self.total_count,
                'correct':  self.correct_count,
                'accuracy': accuracy,
                'weights':  {'W_REL': W_REL, 'W_SUP': W_SUP, 'W_USE': W_USE},
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
            'results': self.results,
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[Save] Results saved to: {filename}")
        return filename

    def run(self):
        data = self.load_data()
        print(f"\n{'='*60}")
        print(f" Self-RAG Baseline — PubMedQA Evaluation")
        print(f" Total questions: {len(data)}")
        print(f" Weights: W_REL={W_REL}, W_SUP={W_SUP}, W_USE={W_USE}")
        print(f"{'='*60}\n")

        for i, item in enumerate(tqdm(data, desc='PubMedQA')):
            print(f"\n--- Q {i+1}/{len(data)} (pmid={item.get('pmid')}) ---")
            result = self.run_single(item)

            self.total_count += 1
            if result['correct']:
                self.correct_count += 1
            self.results.append(result)

            acc = self.correct_count / self.total_count
            print(f"  GT={result['gt']}, Pred={result['pred']}, "
                  f"Correct={result['correct']}  |  "
                  f"Running Acc={acc:.3f} ({self.correct_count}/{self.total_count})")

            if (i + 1) % PUBMEDQA_SAVE_INTERVAL == 0:
                self.save_results()

        path = self.save_results()
        print(f"\n{'='*60}")
        print(f" FINAL ACCURACY: {self.correct_count}/{self.total_count} "
              f"= {self.correct_count/self.total_count:.4f}")
        print(f" Results saved: {path}")
        print(f"{'='*60}")
        return self.correct_count / self.total_count


# ============================================================
#  MedQA 评估器
# ============================================================

# ---- 数据集路径（可修改）----
MEDQA_DATA_PATH  = "data/medqa_sample.jsonl"
MEDQA_OUTPUT_DIR = "TEST_RESULTS/self_rag/medqa"

# ---- 测试数量（None = 全量）----
MEDQA_TEST_LIMIT = None

# ---- 保存间隔 ----
MEDQA_SAVE_INTERVAL = 50


class MedQASelfRAGEvaluator:
    """
    在 MedQA 数据集上评估 Self-RAG Baseline。
    数据集格式: JSONL，每行 {question, options:{A,B,C,D}, answer}
    """

    def __init__(self):
        self.pipeline    = SelfRAGPipeline(task_type='medqa')
        self.results: List[Dict] = []
        self.correct_count = 0
        self.total_count   = 0
        os.makedirs(MEDQA_OUTPUT_DIR, exist_ok=True)

    def load_data(self) -> List[Dict]:
        data = []
        with open(MEDQA_DATA_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if MEDQA_TEST_LIMIT is not None and i >= MEDQA_TEST_LIMIT:
                    break
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    @staticmethod
    def normalize_answer(ans: str) -> str:
        ans = (ans or '').strip().upper()
        if ans and ans[0] in 'ABCD':
            return ans[0]
        return ''

    def run_single(self, item: Dict) -> Dict:
        question     = item.get('question', '')
        options      = item.get('options', {})
        # MedQA 数据集中答案字母存放在 answer_idx 字段
        # answer 字段是完整文本（如 "Tell the attending..."），不能直接用首字符判断
        ground_truth = self.normalize_answer(
            item.get('answer_idx', '') or item.get('answer', '')
        )

        try:
            result    = self.pipeline.run(question, options=options)
            predicted = self.normalize_answer(result['answer'])
            correct   = (predicted == ground_truth) and bool(predicted)

            return {
                'question':   question,
                'options':    options,
                'gt':         ground_truth,
                'pred':       predicted,
                'correct':    correct,
                'retrieved':  result['retrieved'],
                'best_score': result['best']['score'],
                'best_isrel': result['best']['isrel'],
                'best_issup': result['best']['issup'],
                'best_isuse': result['best']['isuse'],
                'raw_answer': result['best']['raw'],
                'error':      None,
            }
        except Exception as e:
            return {
                'question': question, 'options': options,
                'gt': ground_truth, 'pred': '', 'correct': False,
                'retrieved': False, 'best_score': 0.0,
                'best_isrel': None, 'best_issup': None, 'best_isuse': None,
                'raw_answer': '', 'error': str(e),
            }

    def save_results(self, filename: Optional[str] = None):
        if filename is None:
            ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(MEDQA_OUTPUT_DIR,
                                    f'self_rag_medqa_{ts}.json')
        accuracy = self.correct_count / self.total_count if self.total_count else 0
        output   = {
            'meta': {
                'total':    self.total_count,
                'correct':  self.correct_count,
                'accuracy': accuracy,
                'weights':  {'W_REL': W_REL, 'W_SUP': W_SUP, 'W_USE': W_USE},
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
            'results': self.results,
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[Save] Results saved to: {filename}")
        return filename

    def run(self):
        data = self.load_data()
        print(f"\n{'='*60}")
        print(f" Self-RAG Baseline — MedQA Evaluation")
        print(f" Total questions: {len(data)}")
        print(f" Weights: W_REL={W_REL}, W_SUP={W_SUP}, W_USE={W_USE}")
        print(f"{'='*60}\n")

        for i, item in enumerate(tqdm(data, desc='MedQA')):
            print(f"\n--- Q {i+1}/{len(data)} ---")
            result = self.run_single(item)

            self.total_count += 1
            if result['correct']:
                self.correct_count += 1
            self.results.append(result)

            acc = self.correct_count / self.total_count
            print(f"  GT={result['gt']}, Pred={result['pred']}, "
                  f"Correct={result['correct']}  |  "
                  f"Running Acc={acc:.3f} ({self.correct_count}/{self.total_count})")

            if (i + 1) % MEDQA_SAVE_INTERVAL == 0:
                self.save_results()

        path = self.save_results()
        print(f"\n{'='*60}")
        print(f" FINAL ACCURACY: {self.correct_count}/{self.total_count} "
              f"= {self.correct_count/self.total_count:.4f}")
        print(f" Results saved: {path}")
        print(f"{'='*60}")
        return self.correct_count / self.total_count


# ============================================================
#  单问题演示 / 快速测试
# ============================================================

def demo_single_question():
    """
    使用 Mock 文档演示 SELF-RAG 完整推理流程（不实际调用 PubMed）。
    适合验证流程逻辑与 LLM 输出格式是否正确。
    """
    print("\n" + "="*65)
    print(" SELF-RAG DEMO — Single Question with Mock Documents")
    print("="*65)

    # ── 测试问题（生物医学领域复杂问题）──────────────────────────
    question = (
        "Explain the interaction between the BRCA1 gene and PARP inhibitors "
        "in the context of cancer therapy. What is the mechanism of synthetic "
        "lethality, and which tumor types benefit most from this strategy?"
    )

    # ── Mock 检索文档（模拟 PubMed 返回结果）──────────────────────
    mock_docs = [
        # 文档 1: 高相关性
        (
            "Title: PARP Inhibitors and BRCA-Mutated Cancers: Mechanism and Clinical Outcomes\n"
            "Abstract: PARP inhibitors (PARPi) exploit the concept of synthetic lethality in "
            "BRCA1/2-mutated tumors. BRCA1 and BRCA2 are key mediators of homologous "
            "recombination (HR) repair. Cancer cells with BRCA mutations rely on PARP-mediated "
            "base-excision repair (BER) for DNA damage tolerance. When PARP is inhibited, "
            "single-strand breaks accumulate, collapse into double-strand breaks (DSBs), "
            "and HR-deficient BRCA-mutant cells cannot repair them — leading to cell death. "
            "Olaparib, niraparib, and rucaparib have shown significant efficacy in BRCA-mutant "
            "ovarian and breast cancers in Phase III trials."
        ),
        # 文档 2: 部分相关
        (
            "Title: DNA Damage Response Pathways and Targeted Therapy\n"
            "Abstract: The DNA damage response (DDR) is a complex network of pathways that "
            "maintain genomic stability. Homologous recombination (HR) and non-homologous "
            "end-joining (NHEJ) are the two primary DSB repair mechanisms. Defects in HR, "
            "often caused by BRCA1/2 mutations, sensitize cells to DSB-inducing agents. "
            "Synthetic lethality occurs when two gene defects together cause cell death, "
            "but either defect alone is tolerated. This principle underpins PARPi therapy."
        ),
        # 文档 3: 低相关性
        (
            "Title: Epidemiology of Hereditary Breast and Ovarian Cancer Syndrome\n"
            "Abstract: BRCA1 and BRCA2 mutations account for the majority of hereditary breast "
            "and ovarian cancer (HBOC) syndrome cases. Lifetime risk of breast cancer in BRCA1 "
            "carriers is approximately 65-72%, while ovarian cancer risk is 39-46%. Genetic "
            "counseling and testing are recommended for high-risk families. Prophylactic "
            "surgeries (mastectomy, salpingo-oophorectomy) reduce cancer risk significantly."
        ),
    ]

    # ── 使用 Mock 文档绕过实际检索 ────────────────────────────────
    class DemoRetriever(SelfRAGRetriever):
        """返回预设 Mock 文档，不访问网络"""
        def retrieve(self, query: str) -> List[str]:
            print(f"  [MockRetriever] Returning {len(mock_docs)} mock documents")
            return mock_docs

    pipeline = SelfRAGPipeline(
        task_type='general',
        retriever=DemoRetriever(top_k=3)
    )

    print(f"\nQuestion: {question}\n")

    # ── Step 1: 检索决策 ──────────────────────────────────────────
    print("─── Step 1: Retrieval Decision ─────────────────────────")
    result = pipeline.run(question)

    # ── 展示结果 ─────────────────────────────────────────────────
    print("\n─── Final Result ───────────────────────────────────────")
    print(f"Retrieved: {result['retrieved']}")
    print(f"Best Score: {result['best']['score']:.4f}")
    print(f"ISREL: {result['best']['isrel']}")
    print(f"ISSUP: {result['best']['issup']}")
    print(f"ISUSE: {result['best']['isuse']}")
    print(f"\nAnswer Segment:\n{result['best']['segment'][:500]}")

    if len(result['candidates']) > 1:
        print("\n─── All Candidates Scores ──────────────────────────────")
        for i, c in enumerate(result['candidates']):
            print(f"  Candidate {i+1}: score={c['score']:.4f}, "
                  f"ISREL={c['isrel']}, ISSUP={c['issup']}, ISUSE={c['isuse']}")

    return result


# ============================================================
#  命令行入口
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='SELF-RAG Baseline Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 运行 Demo（单问题演示，含 Mock 文档）:
  python self_rag_baseline.py --mode demo

  # 在 PubMedQA 上评估:
  python self_rag_baseline.py --mode pubmedqa --limit 100

  # 在 MedQA 上评估:
  python self_rag_baseline.py --mode medqa --limit 50

  # 修改评分权重:
  python self_rag_baseline.py --mode pubmedqa --w_rel 0.4 --w_sup 0.3 --w_use 0.3
        """
    )
    parser.add_argument('--mode', choices=['demo', 'pubmedqa', 'medqa'],
                        default='demo', help='运行模式 (default: demo)')
    parser.add_argument('--limit', type=int, default=None,
                        help='测试数量上限 (default: 全量)')
    parser.add_argument('--top_k', type=int, default=TOP_K,
                        help=f'检索文档数量 (default: {TOP_K})')
    # 允许通过命令行覆盖全局权重
    parser.add_argument('--w_rel', type=float, default=W_REL,
                        help=f'ISREL 权重 (default: {W_REL})')
    parser.add_argument('--w_sup', type=float, default=W_SUP,
                        help=f'ISSUP 权重 (default: {W_SUP})')
    parser.add_argument('--w_use', type=float, default=W_USE,
                        help=f'ISUSE 权重 (default: {W_USE})')

    args = parser.parse_args()

    # 用命令行参数覆盖全局权重
    W_REL  = args.w_rel
    W_SUP  = args.w_sup
    W_USE  = args.w_use
    TOP_K  = args.top_k

    if args.mode == 'demo':
        demo_single_question()

    elif args.mode == 'pubmedqa':
        if args.limit is not None:
            PUBMEDQA_TEST_LIMIT = args.limit
        evaluator = PubMedQASelfRAGEvaluator()
        evaluator.run()

    elif args.mode == 'medqa':
        if args.limit is not None:
            MEDQA_TEST_LIMIT = args.limit
        evaluator = MedQASelfRAGEvaluator()
        evaluator.run()
