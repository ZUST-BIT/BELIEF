"""
MEDAR-QA 统一配置模块
所有配置通过环境变量加载，支持 .env 文件覆盖默认值。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


def _bool(v: str | None, default: bool = False) -> bool:
    """解析布尔型环境变量"""
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(v: str | None, default: int) -> int:
    """解析整型环境变量"""
    if v is None or v == "":
        return default
    return int(v)


# ==================== LLM 后端配置 ====================
LLM_BACKEND = os.getenv("MEDAR_LLM_BACKEND", "ollama")

# OpenAI 兼容 API
API_URL = os.getenv("MEDAR_API_URL", "https://api.gptsapi.net/v1")
API_KEY = os.getenv("MEDAR_API_KEY", "")
MODEL_NAME = os.getenv("MEDAR_MODEL_NAME", "gpt-4o-mini")

# Ollama
OLLAMA_URL = os.getenv("MEDAR_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("MEDAR_OLLAMA_MODEL", "qwen2.5:7b")

# vLLM
VLLM_URL = os.getenv("MEDAR_VLLM_URL", "http://localhost:8001/v1")
VLLM_MODEL = os.getenv("MEDAR_VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# 本地 Transformers
LOCAL_MODEL_PATH = os.getenv("MEDAR_LOCAL_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
DEVICE = os.getenv("MEDAR_DEVICE", "cuda")

# ==================== Think 模式控制 ====================
DISABLE_THINKING = _bool(os.getenv("MEDAR_DISABLE_THINKING"), default=True)

# ==================== 模型微调参数 ====================
MODEL_REASONING_LEVELS = {
    "gpt-oss:20b": "high",
}
MODEL_SYSTEM_PROMPTS = {
    "gpt-oss:20b": (
        "You are a biomedical QA assistant. "
        "Prefer concise answers and put final answer first."
    ),
}
MODEL_DISABLE_THINKING_OVERRIDES = {
    "gpt-oss:20b": False,
}

# ==================== 数据库配置 ====================
NEO4J_USR = os.getenv("MEDAR_NEO4J_USR", "neo4j")
NEO4J_PWD = os.getenv("MEDAR_NEO4J_PWD", "")
NEO4J_URL = os.getenv("MEDAR_NEO4J_URL", "bolt://localhost:7687")
MONGO_URL = os.getenv("MEDAR_MONGO_URL", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MEDAR_DB_NAME", "bio")
COLLECTION_NAME = os.getenv("MEDAR_COLLECTION_NAME", "pubmed")

# ==================== FAISS / 向量检索配置 ====================
EMBEDDING_MODEL_ZH = os.getenv("MEDAR_EMBEDDING_MODEL_ZH", "BAAI/bge-large-zh")
FAISS_INDEX_PATH_OMIC = os.getenv("MEDAR_FAISS_INDEX_PATH_OMIC", "output/faiss_index_omic.bin")
FAISS_MAPPING_PATH_OMIC = os.getenv("MEDAR_FAISS_MAPPING_PATH_OMIC", "output/faiss_mapping_omic.json")
EMBEDDING_MODEL_EN = os.getenv("MEDAR_EMBEDDING_MODEL_EN", "BAAI/bge-large-en-v1.5")
FAISS_INDEX_PATH_BIO = os.getenv("MEDAR_FAISS_INDEX_PATH_BIO", "output/faiss_index_bio.bin")
FAISS_MAPPING_PATH_BIO = os.getenv("MEDAR_FAISS_MAPPING_PATH_BIO", "output/faiss_mapping_bio.json")
FAISS_INDEX_PATH_HCC = os.getenv("MEDAR_FAISS_INDEX_PATH_HCC", "output/faiss_index_hcc.bin")
FAISS_MAPPING_PATH_HCC = os.getenv("MEDAR_FAISS_MAPPING_PATH_HCC", "output/faiss_mapping_hcc.json")

# ==================== 数据路径配置 ====================
INPUT_DIR = os.getenv("MEDAR_INPUT_DIR", "D:/BitLabData/bio关键文档/outputs_final")
FACT_JSONL = os.getenv("MEDAR_FACT_JSONL", "datafile/fact_corpus_explanatory.jsonl")
FACT_JSONL_HCC = os.getenv("MEDAR_FACT_JSONL_HCC", "D:/BitLabData/bio关键文档/facts_cache.jsonl")
MESH_MAPPING_PATH = os.getenv("MEDAR_MESH_MAPPING_PATH", "D:/BitLabData/MeSH/mesh_mapping.json")
MESH_INFO_PATH = os.getenv("MEDAR_MESH_INFO_PATH", "D:/BitLabData/MeSH/mesh_info.json")

# ==================== 通用参数 ====================
PROJECT = os.getenv("MEDAR_PROJECT", "BELIEF")
BATCH_SIZE = _int(os.getenv("MEDAR_BATCH_SIZE"), 32)
