"""
RAT (Retrieval Augmented Thoughts) Baseline 实现
=================================================
Paper: "RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning
        in Long-Horizon Generation" (2024)

严格按照论文算法实现因果迭代检索增强思维链，用于在 PubMedQA 和 MedQA
数据集上评估准确率，与 MEDAR-QA 流程做基线对比。

算法流程：
  Step 1 - 零样本 CoT：生成初始推理步骤 T1, T2, ..., Tn
  Step 2 - 因果检索增强修订：
           对每一步 Ti：
             2.1 用 (question, T*_1:i-1, Ti) 构建查询 Qi
             2.2 检索 top-k 文档（FAISS + Sentence-Transformers）
             2.3 仅修订当前步骤 Ti → Ti*
  Step 3 - 用修订后的完整推理链生成最终答案

用法：
  python rat_baseline.py --task pubmedqa --limit 50
  python rat_baseline.py --task medqa    --limit 50
  python rat_baseline.py --task both
"""

import json
import os
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm

# ── 使用项目内已有的 LLM 客户端 ──────────────────────────────────────────────
from llm_client import get_llm_client

# ── 向量检索依赖（可选，缺失时给出提示）───────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False
    print(
        "⚠️  缺少依赖，向量检索不可用。\n"
        "   请执行: pip install sentence-transformers faiss-cpu"
    )

# ==================== 配置参数 ====================
PUBMEDQA_PATH    = "data/pubmedqa_sample.json"
MEDQA_PATH       = "data/medqa_sample.jsonl"
OUTPUT_DIR       = "TEST_RESULTS/rat_baseline"
TEST_LIMIT       = None          # None = 测试全部数据
SAVE_INTERVAL    = 50            # 每隔多少条自动保存一次中间结果
TOP_K            = 3             # 每步检索的文档数量
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"   # 也可使用 "all-MiniLM-L6-v2"
# =================================================


# ─────────────────────────────────────────────────────────────────────────────
#  FAISS 检索器
# ─────────────────────────────────────────────────────────────────────────────

class FAISSRetriever:
    """
    基于 FAISS 的向量检索器
    使用 Sentence-Transformers 生成嵌入，余弦相似度（内积，已归一化）检索
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index = None
        self.docs: List[str] = []

    def _lazy_load_model(self):
        if self.model is None:
            print(f"⏳ 加载嵌入模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("✅ 嵌入模型加载完成")

    def build_index(self, documents: List[str]):
        """构建 FAISS 索引"""
        self._lazy_load_model()
        print(f"⏳ 构建 FAISS 索引，文档数量: {len(documents)}")
        self.docs = documents
        embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            batch_size=128,
            normalize_embeddings=True,   # 归一化后内积 = 余弦相似度
        )
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"✅ 索引构建完成  [维度={dim}，文档数={len(self.docs)}]")

    def search(self, query: str, top_k: int = TOP_K) -> List[str]:
        """根据查询检索最相关的 top_k 文档"""
        if self.index is None or not self.docs:
            return []
        self._lazy_load_model()
        query_emb = self.model.encode(
            [query], normalize_embeddings=True
        )
        query_emb = np.array(query_emb, dtype="float32")
        k = min(top_k, len(self.docs))
        scores, indices = self.index.search(query_emb, k)
        return [self.docs[idx] for idx in indices[0] if idx < len(self.docs)]


# ─────────────────────────────────────────────────────────────────────────────
#  RATReasoner：核心推理类
# ─────────────────────────────────────────────────────────────────────────────

class RATReasoner:
    """
    RAT 推理核心类

    严格实现论文中的三步算法：
      1. generate_initial_cot()  — 零样本 CoT
      2. construct_query()       — 构建因果查询
         retrieve()              — 向量检索
         revise_step()           — 仅修订当前步骤
      3. generate_final_answer() — 基于修订链生成答案
    """

    def __init__(self, llm_client, retriever: FAISSRetriever, top_k: int = TOP_K):
        self.llm = llm_client
        self.retriever = retriever
        self.top_k = top_k

    # ── Step 1 ─────────────────────────────────────────────────────────────

    def generate_initial_cot(
        self,
        question: str,
        options: Optional[str] = None,
    ) -> List[str]:
        """
        零样本 Chain-of-Thought：生成初始推理步骤列表 T1, T2, ..., Tn
        """
        if options:
            prompt = (
                f"Question: {question}\n"
                f"Options:\n{options}\n\n"
                "Let's think step by step.\n"
                'Number each reasoning step clearly as "Step 1:", "Step 2:", etc.'
            )
        else:
            prompt = (
                f"Question: {question}\n\n"
                "Let's think step by step.\n"
                'Number each reasoning step clearly as "Step 1:", "Step 2:", etc.'
            )

        response = self.llm.chat(prompt, temperature=0, max_tokens=1200)
        steps = self._parse_steps(response)
        return steps

    def _parse_steps(self, cot_text: str) -> List[str]:
        """将 CoT 文本解析为步骤列表"""
        # 优先匹配 "Step N:" / "Step N." 格式
        matches = re.findall(
            r"Step\s+\d+[:.]\s*(.+?)(?=Step\s+\d+[:.]\s*|\Z)",
            cot_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if matches and len(matches) >= 2:
            return [m.strip() for m in matches if m.strip()]

        # 降级：按换行或句号分割
        sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", cot_text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if not sentences:
            return [cot_text.strip()]
        # 至少保留 2 步，最多 6 步
        return sentences[:6]

    # ── Step 2.1 ───────────────────────────────────────────────────────────

    def construct_query(
        self,
        question: str,
        revised_steps: List[str],
        current_step: str,
    ) -> str:
        """
        构建检索查询 Qi

        Qi = (原始问题) + (已修订步骤 T*_1:i-1 的最近 2 步) + (当前原始步骤 Ti)
        每步查询都会因上下文变化而不同（causal）。
        """
        parts = [question]
        if revised_steps:
            # 使用最近 2 步已修订内容作为上下文锚点
            recent = " ".join(revised_steps[-2:])
            parts.append(f"Previous reasoning: {recent}")
        parts.append(f"Current reasoning focus: {current_step}")
        return " | ".join(parts)

    # ── Step 2.2 ───────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[str]:
        """从向量数据库检索 top-k 文档"""
        return self.retriever.search(query, self.top_k)

    # ── Step 2.3 ───────────────────────────────────────────────────────────

    def revise_step(
        self,
        question: str,
        options: Optional[str],
        revised_steps: List[str],
        current_step: str,
        retrieved_docs: List[str],
    ) -> str:
        """
        仅修订当前推理步骤（因果增量修订）

        关键约束（论文要求）：
          - 不重写整条推理链
          - 不汇总检索文档
          - 只输出修订后的当前步骤
        """
        docs_text = "\n\n".join(
            [f"[Doc {i+1}]: {doc}" for i, doc in enumerate(retrieved_docs)]
        )
        prev_reasoning = (
            "\n".join([f"Step {i+1}*: {s}" for i, s in enumerate(revised_steps)])
            if revised_steps
            else "(none yet)"
        )

        base_prompt = (
            "You are refining a reasoning chain step-by-step using retrieved evidence.\n\n"
            f"Original Question: {question}\n"
        )
        if options:
            base_prompt += f"Options:\n{options}\n"

        base_prompt += (
            f"\nPreviously revised steps:\n{prev_reasoning}\n\n"
            f"Current original reasoning step to revise:\n{current_step}\n\n"
            f"Retrieved documents:\n{docs_text}\n\n"
            "Task: Revise ONLY the current reasoning step based on the retrieved documents "
            "and the question context.\n"
            "Rules:\n"
            "  - Do NOT rewrite previous steps\n"
            "  - Do NOT generate the final answer yet\n"
            "  - Do NOT merely summarize the documents\n"
            "  - Output ONLY the revised version of this single step (1-3 sentences)\n\n"
            "Revised step:"
        )

        revised = self.llm.chat(base_prompt, temperature=0, max_tokens=400)
        return revised.strip()

    # ── Step 3 ─────────────────────────────────────────────────────────────

    def generate_final_answer(
        self,
        question: str,
        options: Optional[str],
        revised_steps: List[str],
        task_type: str = "mcq",
    ) -> str:
        """
        基于修订后的完整推理链生成最终答案
        task_type: "mcq" (A/B/C/D) | "yesno" (yes/no/maybe)
        """
        reasoning_chain = "\n".join(
            [f"Step {i+1}*: {s}" for i, s in enumerate(revised_steps)]
        )

        if task_type == "yesno":
            prompt = (
                f"Question: {question}\n\n"
                f"Revised reasoning chain:\n{reasoning_chain}\n\n"
                "Based on the revised reasoning above, answer the question "
                "with ONLY one word: yes, no, or maybe.\n\n"
                "Answer:"
            )
        else:
            prompt = (
                f"Question: {question}\n\n"
                f"Options:\n{options}\n\n"
                f"Revised reasoning chain:\n{reasoning_chain}\n\n"
                "Based on the revised reasoning above, select the correct option "
                "and respond with ONLY the letter (A, B, C, or D).\n\n"
                "Answer:"
            )

        response = self.llm.chat(prompt, temperature=0, max_tokens=10)
        return response.strip()

    # ── 完整推理入口 ────────────────────────────────────────────────────────

    def reason(
        self,
        question: str,
        options: Optional[str] = None,
        task_type: str = "mcq",
    ) -> Dict:
        """
        完整 RAT 推理流程

        Returns：
          {
            "initial_steps":       List[str],   原始 CoT 步骤
            "revised_steps":       List[str],   修订后步骤
            "retrieved_per_step":  List[List[str]],  每步检索的文档
            "final_answer":        str,          最终答案
          }
        """
        # ── Step 1：零样本 CoT ──────────────────────────────────────────────
        initial_steps = self.generate_initial_cot(question, options)

        # ── Step 2：因果迭代检索增强修订 ────────────────────────────────────
        revised_steps: List[str] = []
        all_retrieved: List[List[str]] = []

        for step in initial_steps:
            # 2.1 构建查询（每步不同）
            query = self.construct_query(question, revised_steps, step)
            # 2.2 检索
            docs = self.retrieve(query)
            all_retrieved.append(docs)
            # 2.3 仅修订当前步骤
            revised = self.revise_step(question, options, revised_steps, step, docs)
            revised_steps.append(revised)

        # ── Step 3：最终答案 ────────────────────────────────────────────────
        final_answer = self.generate_final_answer(
            question, options, revised_steps, task_type
        )

        return {
            "initial_steps": initial_steps,
            "revised_steps": revised_steps,
            "retrieved_per_step": all_retrieved,
            "final_answer": final_answer,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  数据加载工具
# ─────────────────────────────────────────────────────────────────────────────

def load_pubmedqa(path: str, limit: Optional[int] = None) -> List[Dict]:
    """加载 PubMedQA 数据集（pubmedqa_hard.json）"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = []
    for pmid, entry in raw.items():
        items.append(
            {
                "id": pmid,
                "question": entry["QUESTION"],
                "contexts": entry.get("CONTEXTS", []),
                "final_decision": entry["final_decision"],   # yes / no / maybe
            }
        )
        if limit and len(items) >= limit:
            break
    return items


def load_medqa(path: str, limit: Optional[int] = None) -> List[Dict]:
    """加载 MedQA 数据集（medqa_sample.jsonl）"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
                if limit and len(items) >= limit:
                    break
    return items


def build_pubmedqa_corpus(data: List[Dict]) -> List[str]:
    """
    从 PubMedQA 全数据集的 CONTEXTS 字段构建检索语料库
    这模拟了 RAT 论文中使用外部知识库的设置
    """
    corpus = []
    for item in data:
        for ctx in item.get("contexts", []):
            if ctx and len(ctx.strip()) > 20:
                corpus.append(ctx.strip())
    # 去重
    return list(dict.fromkeys(corpus))


def build_medqa_corpus(data: List[Dict]) -> List[str]:
    """
    从 MedQA 全数据集的问题+选项构建检索语料库
    充当 RAT 的内部知识检索来源
    """
    corpus = []
    for item in data:
        q = item.get("question", "").strip()
        opts = item.get("options", {})
        # 将每个选项独立作为一条文档，前缀加上问题背景
        for k, v in opts.items():
            if v and len(v.strip()) > 5:
                corpus.append(f"{q} {v.strip()}")
    return list(dict.fromkeys(corpus))


# ─────────────────────────────────────────────────────────────────────────────
#  答案提取工具
# ─────────────────────────────────────────────────────────────────────────────

def extract_yesno(response: str) -> str:
    """从响应中提取 yes / no / maybe"""
    if not response:
        return ""
    r = response.lower().strip()
    # 前缀优先匹配
    for kw in ("yes", "no", "maybe"):
        if r.startswith(kw):
            return kw
    # 任意位置匹配
    for kw in ("yes", "no", "maybe"):
        if kw in r:
            return kw
    return ""


def extract_mcq(response: str) -> str:
    """从响应中提取 A / B / C / D"""
    if not response:
        return ""
    r = response.strip().upper()
    if r and r[0] in "ABCD":
        return r[0]
    for ch in r:
        if ch in "ABCD":
            return ch
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  评估器
# ─────────────────────────────────────────────────────────────────────────────

class RATEvaluator:
    """RAT 基线评估器，支持 pubmedqa 和 medqa 两个任务"""

    def __init__(self, task: str = "pubmedqa"):
        assert task in ("pubmedqa", "medqa"), \
            f"task 必须是 'pubmedqa' 或 'medqa'，收到: {task}"
        self.task = task
        self.llm = get_llm_client()
        self.results: List[Dict] = []
        self.correct = 0
        self.total = 0
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _build_reasoner(self, data: List[Dict]) -> RATReasoner:
        """按任务类型构建 FAISS 索引并实例化 RATReasoner"""
        retriever = FAISSRetriever(EMBEDDING_MODEL)
        if self.task == "pubmedqa":
            corpus = build_pubmedqa_corpus(data)
        else:
            corpus = build_medqa_corpus(data)

        if not corpus:
            raise ValueError("语料库为空，无法构建 FAISS 索引！")

        retriever.build_index(corpus)
        return RATReasoner(self.llm, retriever, TOP_K)

    # ── PubMedQA 评估 ──────────────────────────────────────────────────────

    def run_pubmedqa(self):
        data = load_pubmedqa(PUBMEDQA_PATH, TEST_LIMIT)
        print(f"\n{'='*80}")
        print("RAT Baseline 评估 — PubMedQA")
        print(f"{'='*80}")
        print(f"数据集条数: {len(data)} | TOP_K={TOP_K} | 嵌入模型={EMBEDDING_MODEL}")

        reasoner = self._build_reasoner(data)

        for i, item in enumerate(tqdm(data, desc="PubMedQA RAT")):
            qid = item["id"]
            question = item["question"]
            ground_truth = item["final_decision"].lower().strip()
            initial_steps_out: List[str] = []
            revised_steps_out: List[str] = []
            final_answer_raw = ""

            try:
                result = reasoner.reason(
                    question=question,
                    options=None,
                    task_type="yesno",
                )
                initial_steps_out = result.get("initial_steps", [])
                revised_steps_out = result.get("revised_steps", [])
                final_answer_raw = result.get("final_answer", "")
                predicted = extract_yesno(final_answer_raw)
                is_correct = predicted == ground_truth
                error = None
            except Exception as exc:
                predicted = ""
                is_correct = False
                error = str(exc)
                print(f"\n  ❌ 处理 {qid} 时出错: {exc}")

            self.total += 1
            if is_correct:
                self.correct += 1

            self.results.append(
                {
                    "id": qid,
                    "question": question,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "initial_steps": initial_steps_out,
                    "revised_steps": revised_steps_out,
                    "final_answer_raw": final_answer_raw,
                    "error": error,
                }
            )

            acc = self.correct / self.total * 100
            print(
                f"\n  [{i+1}/{len(data)}] {qid}"
                f"  GT={ground_truth}  Pred={predicted}"
                f"  ✓={is_correct}  Acc={acc:.2f}%"
            )

            if (i + 1) % SAVE_INTERVAL == 0:
                self._save(interim=True)

        self._save(interim=False)
        self._print_summary()

    # ── MedQA 评估 ─────────────────────────────────────────────────────────

    def run_medqa(self):
        data = load_medqa(MEDQA_PATH, TEST_LIMIT)
        print(f"\n{'='*80}")
        print("RAT Baseline 评估 — MedQA")
        print(f"{'='*80}")
        print(f"数据集条数: {len(data)} | TOP_K={TOP_K} | 嵌入模型={EMBEDDING_MODEL}")

        reasoner = self._build_reasoner(data)

        for i, item in enumerate(tqdm(data, desc="MedQA RAT")):
            idx = item.get("realidx", i)
            question = item.get("question", "")
            options = item.get("options", {})
            opts_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
            ground_truth = item.get("answer_idx", "").upper()
            initial_steps_out: List[str] = []
            revised_steps_out: List[str] = []
            final_answer_raw = ""

            try:
                result = reasoner.reason(
                    question=question,
                    options=opts_str,
                    task_type="mcq",
                )
                initial_steps_out = result.get("initial_steps", [])
                revised_steps_out = result.get("revised_steps", [])
                final_answer_raw = result.get("final_answer", "")
                predicted = extract_mcq(final_answer_raw)
                is_correct = predicted == ground_truth
                error = None
            except Exception as exc:
                predicted = ""
                is_correct = False
                error = str(exc)
                print(f"\n  ❌ 处理 idx={idx} 时出错: {exc}")

            self.total += 1
            if is_correct:
                self.correct += 1

            self.results.append(
                {
                    "id": idx,
                    "question": question,
                    "options": options,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "initial_steps": initial_steps_out,
                    "revised_steps": revised_steps_out,
                    "final_answer_raw": final_answer_raw,
                    "error": error,
                }
            )

            acc = self.correct / self.total * 100
            print(
                f"\n  [{i+1}/{len(data)}] idx={idx}"
                f"  GT={ground_truth}  Pred={predicted}"
                f"  ✓={is_correct}  Acc={acc:.2f}%"
            )

            if (i + 1) % SAVE_INTERVAL == 0:
                self._save(interim=True)

        self._save(interim=False)
        self._print_summary()

    # ── 辅助方法 ───────────────────────────────────────────────────────────

    def _save(self, interim: bool = False):
        """保存评估结果到 JSON 文件"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "interim_" if interim else "final_"
        filename = f"{OUTPUT_DIR}/rat_{self.task}_{prefix}{ts}.json"

        payload = {
            "meta": {
                "timestamp": ts,
                "task": self.task,
                "total": self.total,
                "correct": self.correct,
                "accuracy": round(self.correct / self.total * 100, 4) if self.total else 0.0,
                "test_limit": TEST_LIMIT,
                "top_k": TOP_K,
                "embedding_model": EMBEDDING_MODEL,
            },
            "results": self.results,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 已保存到: {filename}")

    def _print_summary(self):
        total = self.total
        correct = self.correct
        acc = correct / total * 100 if total else 0.0

        print(f"\n{'='*80}")
        print(f"RAT Baseline 评估摘要 — {self.task.upper()}")
        print(f"{'='*80}")
        print(f"  总测试数量:  {total}")
        print(f"  正确数量:    {correct}")
        print(f"  准确率:      {acc:.2f}%")

        if self.task == "pubmedqa":
            for label in ("yes", "no", "maybe"):
                n_correct = sum(
                    1 for r in self.results
                    if r["ground_truth"] == label and r["is_correct"]
                )
                n_total = sum(
                    1 for r in self.results if r["ground_truth"] == label
                )
                if n_total:
                    print(
                        f"  [{label:7s}]: {n_correct}/{n_total}"
                        f"  ({n_correct/n_total*100:.2f}%)"
                    )
        else:  # medqa
            for opt in "ABCD":
                n_correct = sum(
                    1 for r in self.results
                    if r["ground_truth"] == opt and r["is_correct"]
                )
                n_total = sum(
                    1 for r in self.results if r["ground_truth"] == opt
                )
                if n_total:
                    print(
                        f"  [Option {opt}]: {n_correct}/{n_total}"
                        f"  ({n_correct/n_total*100:.2f}%)"
                    )

        n_errors = sum(1 for r in self.results if r.get("error"))
        if n_errors:
            print(f"\n  ⚠️  出错条目数: {n_errors}")

        print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    # 声明在此函数内会修改的全局变量，必须在使用前声明
    global TEST_LIMIT, TOP_K, EMBEDDING_MODEL

    parser = argparse.ArgumentParser(
        description="RAT Baseline Evaluation on PubMedQA & MedQA"
    )
    parser.add_argument(
        "--task",
        choices=["pubmedqa", "medqa", "both"],
        default="both",
        help="评估任务: pubmedqa / medqa / both（默认 both）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="测试数量上限（默认 None = 全部）",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=TOP_K,
        help=f"每步检索的文档数量（默认 {TOP_K}）",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"嵌入模型名称（默认 {EMBEDDING_MODEL}）",
    )
    args = parser.parse_args()

    # 允许命令行覆盖全局配置
    if args.limit is not None:
        TEST_LIMIT = args.limit
    if args.topk:
        TOP_K = args.topk
    if args.embedding_model:
        EMBEDDING_MODEL = args.embedding_model

    if not RETRIEVAL_AVAILABLE:
        print(
            "❌ 向量检索依赖缺失，请执行:\n"
            "   pip install sentence-transformers faiss-cpu"
        )
        return

    print("=" * 80)
    print("RAT Baseline — Retrieval Augmented Thoughts")
    print(f"任务: {args.task}  |  测试限制: {TEST_LIMIT or '全部'}  |  TOP_K: {TOP_K}")
    print(f"嵌入模型: {EMBEDDING_MODEL}")
    print("=" * 80)

    if args.task in ("pubmedqa", "both"):
        evaluator = RATEvaluator(task="pubmedqa")
        evaluator.run_pubmedqa()

    if args.task in ("medqa", "both"):
        evaluator = RATEvaluator(task="medqa")
        evaluator.run_medqa()


if __name__ == "__main__":
    main()
