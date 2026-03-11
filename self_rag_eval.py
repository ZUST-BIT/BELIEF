"""
Self-RAG 评估脚本 —— PubMedQA & MedQA
=========================================
模型: SciPhi Self-RAG Mistral 7B 32K (AWQ)
论文: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
      Asai et al., 2023  https://arxiv.org/abs/2310.11511

使用方法:
  # 测试 PubMedQA (默认 pubmedqa_hard.json)
  python self_rag_eval.py --dataset pubmedqa

  # 测试 MedQA (默认 medqa_sample.jsonl)
  python self_rag_eval.py --dataset medqa

  # 指定本地模型路径 (从 ModelScope 下载后)
  python self_rag_eval.py --dataset pubmedqa --model_path /path/to/model

  # 限制测试条目数
  python self_rag_eval.py --dataset pubmedqa --limit 50

依赖安装:
  pip install autoawq modelscope transformers accelerate
  # 若 AutoAWQ 安装失败，可从源码安装:
  # git clone https://github.com/casper-hansen/AutoAWQ && cd AutoAWQ && pip install .
"""

import os
import json
import re
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# ============================================================
#  日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
#  Self-RAG 特殊 Token 定义（原论文 & SciPhi 实现）
# ============================================================
RETRIEVE_TOKEN       = "[Retrieve]"          # 触发检索
NO_RETRIEVE_TOKEN    = "[No Retrieval]"      # 跳过检索
RELEVANT_TOKEN       = "[Relevant]"          # 文档相关
IRRELEVANT_TOKEN     = "[Irrelevant]"        # 文档无关
DOC_START_TOKEN      = "[Doc]"              # 文档起始
DOC_END_TOKEN        = "[/Doc]"             # 文档结束
FULLY_SUP_TOKEN      = "[Fully supported]"  # 完全支持
PARTIAL_SUP_TOKEN    = "[Partially supported]"
NO_SUP_TOKEN         = "[No support / Contradictory]"
UTILITY_TOKENS       = [f"[Utility:{i}]" for i in range(1, 6)]  # [Utility:1]..[Utility:5]

# SciPhi 提示模板
SYSTEM_MESSAGE = (
    "You are a helpful, respectful and honest medical question answering assistant. "
    "Always answer as helpfully as possible, while being safe."
)

def build_prompt(
    question: str,
    contexts: Optional[list[str]] = None,
    system: str = SYSTEM_MESSAGE,
    inject_retrieve: bool = True,
) -> str:
    """
    构建 Self-RAG 格式的推理 Prompt。

    若 contexts 不为空，则将其格式化为 [Doc]...[/Doc] 拼接到 Response 前缀中，
    让模型在"已检索"前提下继续生成反思与答案。
    若 contexts 为空，则在 Response 前注入 [No Retrieval] 让模型直接回答。
    """
    prompt = f"### System:\n{system}\n\n### Instruction:\n{question}\n\n### Response:\n"

    if contexts:
        docs = ""
        for i, ctx in enumerate(contexts[:3], 1):   # 最多取前 3 段避免超长
            ctx_clean = ctx.strip().replace("\n", " ")[:600]  # 每段截断 600 字符
            docs += f"{DOC_START_TOKEN}{ctx_clean}{DOC_END_TOKEN}"
        if inject_retrieve:
            prompt += f"{RETRIEVE_TOKEN} {docs}"
    else:
        prompt += f"{NO_RETRIEVE_TOKEN} "

    return prompt


# ============================================================
#  答案提取工具
# ============================================================
PUBMEDQA_LABELS = {"yes", "no", "maybe"}

def extract_pubmedqa_answer(text: str) -> Optional[str]:
    """
    从 Self-RAG 生成文本中提取 yes/no/maybe。
    策略（按优先级）：
      1. 开头显式说明 (The answer is yes/no/maybe)
      2. 文本中最先出现 yes/no/maybe（忽略大小写）
      3. 返回 None
    """
    text_lower = text.lower()

    # 优先匹配明确表述
    explicit = re.search(
        r"\b(the answer is|answer:|final answer[:\s]+|decision[:\s]+)\s*(yes|no|maybe)\b",
        text_lower,
    )
    if explicit:
        return explicit.group(2)

    # 退回到首次出现
    match = re.search(r"\b(yes|no|maybe)\b", text_lower)
    if match:
        return match.group(1)
    return None


def extract_medqa_answer(text: str) -> Optional[str]:
    """
    从 Self-RAG 生成文本中提取选项字母 A/B/C/D。
    策略（按优先级）：
      1. "The answer is (A)" / "answer: A" 等明确格式
      2. 单独出现的 (A)/(B)/(C)/(D)
      3. 文本开头首字母
    """
    text_stripped = text.strip()

    # 明确格式
    explicit = re.search(
        r"(?:the\s+answer\s+is|answer[:\s]+|correct\s+answer[:\s]+)\s*\(?([A-Da-d])\)?",
        text_stripped,
        re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1).upper()

    # (A) (B) 格式
    paren = re.search(r"\(([A-Da-d])\)", text_stripped)
    if paren:
        return paren.group(1).upper()

    # 首字母（若行首是选项字母）
    first = re.match(r"^([A-Da-d])[\.:\)\s]", text_stripped)
    if first:
        return first.group(1).upper()

    return None


def strip_reflection_tokens(text: str) -> str:
    """移除 Self-RAG 反思 Token，保留实际回答文本。"""
    for token in (
        [RETRIEVE_TOKEN, NO_RETRIEVE_TOKEN, RELEVANT_TOKEN, IRRELEVANT_TOKEN,
         DOC_START_TOKEN, DOC_END_TOKEN, FULLY_SUP_TOKEN, PARTIAL_SUP_TOKEN, NO_SUP_TOKEN]
        + UTILITY_TOKENS
    ):
        text = text.replace(token, "")
    # 移除 [Doc]...[/Doc] 区块内残余内容
    text = re.sub(r"\[Doc\].*?\[/Doc\]", "", text, flags=re.DOTALL)
    return text.strip()


# ============================================================
#  Self-RAG 模型封装
# ============================================================
class SelfRAGModel:
    """
    封装 SciPhi Self-RAG Mistral 7B 32K (AWQ) 的推理接口。

    支持两种加载方式:
      · AutoAWQ  (推荐，量化 AWQ 模型)
      · Hugging Face Transformers (需要 fp16 原始模型)
    """

    DEFAULT_MODELSCOPE_ID = "SciPhi/SciPhi-Self-RAG-Mistral-7B-32k"
    # ModelScope 上的 AWQ 量化版本（TheBloke 移植）
    DEFAULT_AWQ_ID        = "TheBloke/SciPhi-Self-RAG-Mistral-7B-32k-AWQ"

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_awq: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        device: str = "auto",
    ):
        self.model_path    = model_path or self.DEFAULT_AWQ_ID
        self.use_awq       = use_awq
        self.max_new_tokens = max_new_tokens
        self.temperature   = temperature
        self.top_p         = top_p
        self.device        = device
        self.model         = None
        self.tokenizer     = None

    # ----------------------------------------------------------
    def download_from_modelscope(self, model_id: str) -> str:
        """从 ModelScope 下载模型到本地，返回本地路径。"""
        try:
            from modelscope import snapshot_download
            logger.info(f"正在从 ModelScope 下载模型: {model_id} ...")
            local_path = snapshot_download(model_id)
            logger.info(f"模型已下载到: {local_path}")
            return local_path
        except ImportError:
            raise ImportError(
                "请安装 modelscope: pip install modelscope"
            )

    # ----------------------------------------------------------
    def load(self):
        """加载模型与分词器。"""
        model_path = self.model_path

        # 若传入的是 ModelScope / HuggingFace 模型 ID（非本地路径），尝试下载
        if not Path(model_path).exists():
            logger.info(f"本地路径 {model_path} 不存在，尝试从 ModelScope/HuggingFace 下载...")
            try:
                model_path = self.download_from_modelscope(model_path)
            except Exception as e:
                logger.warning(f"ModelScope 下载失败 ({e})，尝试直接使用 HuggingFace Hub ID...")
                # 保持原始 model_path，让 transformers 自动从 HF Hub 下载

        if self.use_awq:
            self._load_awq(model_path)
        else:
            self._load_transformers(model_path)

    # ----------------------------------------------------------
    def _load_awq(self, model_path: str):
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "请安装 AutoAWQ:\n"
                "  pip install autoawq\n"
                "或从源码安装:\n"
                "  git clone https://github.com/casper-hansen/AutoAWQ && cd AutoAWQ && pip install ."
            )

        logger.info(f"[AWQ] 加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )
        self.model = AutoAWQForCausalLM.from_quantized(
            model_path,
            fuse_layers=True,
            trust_remote_code=False,
            safetensors=True,
        )
        logger.info("[AWQ] 模型加载完成")
        self._backend = "awq"

    # ----------------------------------------------------------
    def _load_transformers(self, model_path: str):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("请安装 transformers 和 torch")

        logger.info(f"[Transformers] 加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=self.device,
            trust_remote_code=False,
        )
        logger.info("[Transformers] 模型加载完成")
        self._backend = "transformers"

    # ----------------------------------------------------------
    def generate(self, prompt: str) -> str:
        """
        执行推理，返回模型生成的文本（不含 prompt 前缀）。
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("请先调用 load() 加载模型")

        if self._backend == "awq":
            return self._generate_awq(prompt)
        else:
            return self._generate_transformers(prompt)

    # ----------------------------------------------------------
    def _generate_awq(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        do_sample = self.temperature > 0.0
        gen_kwargs = dict(
            do_sample=do_sample,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p if do_sample else 1.0,
            top_k=40 if do_sample else 1,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self.model.generate(inputs, **gen_kwargs)

        # 解码，去掉 prompt 部分
        full_text   = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # 取 "### Response:\n" 之后的内容
        response    = self._extract_response(prompt, full_text)
        return response

    # ----------------------------------------------------------
    def _generate_transformers(self, prompt: str) -> str:
        import torch

        inputs     = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len  = inputs["input_ids"].shape[1]

        do_sample = self.temperature > 0.0
        gen_kwargs = dict(
            do_sample=do_sample,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p if do_sample else 1.0,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_ids   = output_ids[0][input_len:]
        response  = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return response.strip()

    # ----------------------------------------------------------
    @staticmethod
    def _extract_response(prompt: str, full_text: str) -> str:
        """从完整解码文本中提取 Response 部分。"""
        marker = "### Response:"
        idx    = full_text.rfind(marker)
        if idx != -1:
            return full_text[idx + len(marker):].strip()
        # 退回到去掉 prompt 前缀
        if full_text.startswith(prompt):
            return full_text[len(prompt):].strip()
        return full_text.strip()


# ============================================================
#  数据加载工具
# ============================================================
def load_pubmedqa(path: str, limit: Optional[int] = None) -> list[dict]:
    """
    加载 PubMedQA JSON 文件。
    期望格式: { pmid: { QUESTION, CONTEXTS, LABELS, final_decision, ... } }
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    items = []
    for pmid, entry in raw.items():
        items.append({
            "id":             pmid,
            "question":       entry.get("QUESTION", ""),
            "contexts":       entry.get("CONTEXTS", []),
            "labels":         entry.get("LABELS", []),
            "final_decision": entry.get("final_decision", "").lower().strip(),
            "long_answer":    entry.get("LONG_ANSWER", ""),
        })
        if limit and len(items) >= limit:
            break

    logger.info(f"加载 PubMedQA: {len(items)} 条  ({path})")
    return items


def load_medqa(path: str, limit: Optional[int] = None) -> list[dict]:
    """
    加载 MedQA JSONL 文件。
    期望字段: question, options{A,B,C,D}, answer_idx, answer
    """
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            options = entry.get("options", {})
            option_text = "\n".join(
                f"  {k}. {v}" for k, v in sorted(options.items())
            )
            question_full = f"{entry['question']}\n{option_text}"
            items.append({
                "id":         str(entry.get("realidx", len(items))),
                "question":   question_full,
                "answer_idx": entry.get("answer_idx", "").upper().strip(),
                "answer":     entry.get("answer", ""),
                "options":    options,
            })
            if limit and len(items) >= limit:
                break

    logger.info(f"加载 MedQA: {len(items)} 条  ({path})")
    return items


# ============================================================
#  评估主逻辑
# ============================================================
class SelfRAGEvaluator:
    """对 PubMedQA 或 MedQA 进行批量评估。"""

    def __init__(
        self,
        model: SelfRAGModel,
        output_dir: str = "TEST_RESULTS/self_rag",
        save_interval: int = 20,
    ):
        self.model         = model
        self.output_dir    = Path(output_dir)
        self.save_interval = save_interval
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    def evaluate_pubmedqa(self, items: list[dict]) -> dict:
        """
        对 PubMedQA 进行逐条评估。
        利用 CONTEXTS 字段作为检索结果注入 Self-RAG 提示。
        """
        results   = []
        correct   = 0
        total     = 0
        skipped   = 0

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"selfrag_pubmedqa_{ts}.json"

        for idx, item in enumerate(items, 1):
            pmid      = item["id"]
            question  = item["question"]
            contexts  = item["contexts"]
            gold      = item["final_decision"]

            if gold not in PUBMEDQA_LABELS:
                skipped += 1
                continue

            prompt    = build_prompt(question, contexts)
            start_t   = time.time()
            try:
                raw_output = self.model.generate(prompt)
            except Exception as e:
                logger.error(f"[{idx}/{len(items)}] pmid={pmid} 推理报错: {e}")
                results.append({
                    "id": pmid, "question": question[:100],
                    "gold": gold, "predicted": None,
                    "correct": False, "raw_output": str(e),
                })
                total += 1
                continue

            elapsed   = time.time() - start_t
            clean_out = strip_reflection_tokens(raw_output)
            pred      = extract_pubmedqa_answer(clean_out)
            is_correct = pred == gold

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "id":          pmid,
                "question":    question[:120],
                "gold":        gold,
                "predicted":   pred,
                "correct":     is_correct,
                "raw_output":  raw_output[:500],
                "clean_output": clean_out[:300],
                "elapsed_s":   round(elapsed, 2),
            })

            tag = "✓" if is_correct else "✗"
            logger.info(
                f"[{idx:>4}/{len(items)}] {tag} gold={gold:<5} pred={str(pred):<5} "
                f"({elapsed:.1f}s)  pmid={pmid}"
            )

            # 定期保存
            if idx % self.save_interval == 0:
                self._save(save_path, results, correct, total, "pubmedqa")

        # 最终保存
        self._save(save_path, results, correct, total, "pubmedqa")
        acc = correct / total if total > 0 else 0.0
        logger.info(
            f"\n{'='*55}\n"
            f"  PubMedQA 评估完成\n"
            f"  总数: {total}  正确: {correct}  准确率: {acc:.4f} ({acc*100:.2f}%)\n"
            f"  跳过(标签缺失): {skipped}\n"
            f"  结果已保存: {save_path}\n"
            f"{'='*55}"
        )
        return {"accuracy": acc, "correct": correct, "total": total, "save_path": str(save_path)}

    # ----------------------------------------------------------
    def evaluate_medqa(self, items: list[dict]) -> dict:
        """
        对 MedQA 进行逐条评估。
        MedQA 为 MCQ，无预设检索内容，使用 [No Retrieval] 模式。
        """
        results   = []
        correct   = 0
        total     = 0

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"selfrag_medqa_{ts}.json"

        for idx, item in enumerate(items, 1):
            qid      = item["id"]
            question = item["question"]
            gold     = item["answer_idx"]

            # 不提供 contexts → No Retrieval 模式
            prompt = build_prompt(question, contexts=None)
            start_t = time.time()
            try:
                raw_output = self.model.generate(prompt)
            except Exception as e:
                logger.error(f"[{idx}/{len(items)}] id={qid} 推理报错: {e}")
                results.append({
                    "id": qid, "question": question[:100],
                    "gold": gold, "predicted": None,
                    "correct": False, "raw_output": str(e),
                })
                total += 1
                continue

            elapsed   = time.time() - start_t
            clean_out = strip_reflection_tokens(raw_output)
            pred      = extract_medqa_answer(clean_out)
            is_correct = pred == gold

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "id":           qid,
                "question":     question[:120],
                "gold":         gold,
                "predicted":    pred,
                "correct":      is_correct,
                "raw_output":   raw_output[:500],
                "clean_output": clean_out[:300],
                "elapsed_s":    round(elapsed, 2),
            })

            tag = "✓" if is_correct else "✗"
            logger.info(
                f"[{idx:>4}/{len(items)}] {tag} gold={gold}  pred={str(pred)}  "
                f"({elapsed:.1f}s)  id={qid}"
            )

            if idx % self.save_interval == 0:
                self._save(save_path, results, correct, total, "medqa")

        self._save(save_path, results, correct, total, "medqa")
        acc = correct / total if total > 0 else 0.0
        logger.info(
            f"\n{'='*55}\n"
            f"  MedQA 评估完成\n"
            f"  总数: {total}  正确: {correct}  准确率: {acc:.4f} ({acc*100:.2f}%)\n"
            f"  结果已保存: {save_path}\n"
            f"{'='*55}"
        )
        return {"accuracy": acc, "correct": correct, "total": total, "save_path": str(save_path)}

    # ----------------------------------------------------------
    def _save(self, path: Path, results: list, correct: int, total: int, dataset: str):
        acc = correct / total if total > 0 else 0.0
        payload = {
            "meta": {
                "dataset":   dataset,
                "timestamp": datetime.now().isoformat(),
                "total":     total,
                "correct":   correct,
                "accuracy":  round(acc, 4),
                "model":     self.model.model_path,
            },
            "results": results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# ============================================================
#  便捷：使用 vLLM API 替代本地加载（可选）
# ============================================================
class SelfRAGvLLMClient:
    """
    通过 vLLM OpenAI 兼容 API 调用已部署的 Self-RAG 模型。

    vLLM 启动命令示例:
      python -m vllm.entrypoints.openai.api_server \
          --model TheBloke/SciPhi-Self-RAG-Mistral-7B-32k-AWQ \
          --quantization awq --port 8001

    用法:
      client = SelfRAGvLLMClient(base_url="http://localhost:8001/v1")
      evaluator = SelfRAGEvaluator(model=client)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model_name: str = "TheBloke/SciPhi-Self-RAG-Mistral-7B-32k-AWQ",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        api_key: str = "EMPTY",
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        self.client        = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name    = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature   = temperature
        self.model_path    = model_name   # 用于报告

    def generate(self, prompt: str) -> str:
        """将 prompt 作为原始文本发送给 vLLM completion 接口。"""
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stop=["### Instruction:", "### System:"],
        )
        return response.choices[0].text.strip()


# ============================================================
#  主入口
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-RAG Evaluation on PubMedQA / MedQA",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["pubmedqa", "medqa", "both"],
        default="pubmedqa",
        help="要评估的数据集 (default: pubmedqa)",
    )
    parser.add_argument(
        "--pubmedqa_path",
        default="data/pubmedqa_hard.json",
        help="PubMedQA 数据路径 (default: data/pubmedqa_hard.json)",
    )
    parser.add_argument(
        "--medqa_path",
        default="data/medqa_sample.jsonl",
        help="MedQA 数据路径 (default: data/medqa_sample.jsonl)",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help=(
            "Self-RAG 模型路径或 ModelScope/HuggingFace ID\n"
            "  默认: TheBloke/SciPhi-Self-RAG-Mistral-7B-32k-AWQ\n"
            "  ModelScope 上的本地路径, 例如:\n"
            "    ~/.cache/modelscope/hub/TheBloke/SciPhi-Self-RAG-Mistral-7B-32k-AWQ"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["awq", "transformers", "vllm"],
        default="awq",
        help=(
            "推理后端:\n"
            "  awq          — AutoAWQ (量化模型，推荐)\n"
            "  transformers — HuggingFace Transformers (fp16 原始模型)\n"
            "  vllm         — vLLM OpenAI API (需先启动服务)"
        ),
    )
    parser.add_argument(
        "--vllm_url",
        default="http://localhost:8001/v1",
        help="vLLM API base URL (仅 --backend vllm 时有效, default: http://localhost:8001/v1)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="最大生成 token 数 (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度，0 表示贪心解码 (default: 0.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个数据集最多评估多少条 (default: 全部)",
    )
    parser.add_argument(
        "--output_dir",
        default="TEST_RESULTS/self_rag",
        help="结果保存目录 (default: TEST_RESULTS/self_rag)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=20,
        help="每多少条保存一次中间结果 (default: 20)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印 prompt 示例，不加载模型（用于调试）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 干运行：仅展示 Prompt 格式 ----
    if args.dry_run:
        print("\n" + "=" * 60)
        print("  DRY RUN —— Self-RAG Prompt 示例")
        print("=" * 60)

        # PubMedQA 示例
        pub_q  = "Does metformin reduce cardiovascular events in type 2 diabetes?"
        pub_ctx = [
            "A randomized controlled trial found that metformin significantly reduced myocardial infarction rates.",
            "Metformin inhibits hepatic gluconeogenesis and activates AMPK.",
        ]
        print("\n[PubMedQA Prompt — with context]")
        print(build_prompt(pub_q, pub_ctx))

        # MedQA 示例
        med_q = (
            "A patient presents with hemoptysis. "
            "Which of the following is the most likely diagnosis?\n"
            "  A. Lung cancer\n  B. Pneumonia\n  C. Tuberculosis\n  D. Asthma"
        )
        print("\n[MedQA Prompt — no context]")
        print(build_prompt(med_q, contexts=None))
        return

    # ---- 加载模型 ----
    if args.backend == "vllm":
        model_name = args.model_path or "TheBloke/SciPhi-Self-RAG-Mistral-7B-32k-AWQ"
        logger.info(f"使用 vLLM 后端: {args.vllm_url}  model={model_name}")
        model = SelfRAGvLLMClient(
            base_url=args.vllm_url,
            model_name=model_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        use_awq = args.backend == "awq"
        model   = SelfRAGModel(
            model_path=args.model_path,
            use_awq=use_awq,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        model.load()

    # ---- 初始化评估器 ----
    evaluator = SelfRAGEvaluator(
        model=model,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
    )

    summary = {}

    # ---- PubMedQA ----
    if args.dataset in ("pubmedqa", "both"):
        items  = load_pubmedqa(args.pubmedqa_path, limit=args.limit)
        result = evaluator.evaluate_pubmedqa(items)
        summary["pubmedqa"] = result

    # ---- MedQA ----
    if args.dataset in ("medqa", "both"):
        items  = load_medqa(args.medqa_path, limit=args.limit)
        result = evaluator.evaluate_medqa(items)
        summary["medqa"] = result

    # ---- 汇总 ----
    print("\n" + "=" * 55)
    print("  最终评估汇总")
    print("=" * 55)
    for k, v in summary.items():
        print(f"  {k:<10} 准确率: {v['accuracy']*100:.2f}%  ({v['correct']}/{v['total']})")
        print(f"           结果文件: {v['save_path']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
