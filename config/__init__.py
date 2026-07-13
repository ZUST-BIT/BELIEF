"""
MEDAR-QA 配置包
向后兼容旧的 config.py 模块，同时提供基于环境变量的统一配置。
"""

import argparse

from .settings import (
    # LLM 后端配置
    LLM_BACKEND,
    API_URL,
    API_KEY,
    MODEL_NAME,
    OLLAMA_URL,
    OLLAMA_MODEL,
    VLLM_URL,
    VLLM_MODEL,
    LOCAL_MODEL_PATH,
    DEVICE,
    # Think 模式控制
    DISABLE_THINKING,
    # PubMed / NCBI E-utilities
    NCBI_TOOL,
    NCBI_EMAIL,
    NCBI_API_KEY,
    # 模型微调参数
    MODEL_REASONING_LEVELS,
    MODEL_SYSTEM_PROMPTS,
    MODEL_DISABLE_THINKING_OVERRIDES,
    # 数据库配置
    NEO4J_USR,
    NEO4J_PWD,
    NEO4J_URL,
    MONGO_URL,
    DB_NAME,
    COLLECTION_NAME,
    # FAISS / 向量检索配置
    EMBEDDING_MODEL_ZH,
    FAISS_INDEX_PATH_OMIC,
    FAISS_MAPPING_PATH_OMIC,
    EMBEDDING_MODEL_EN,
    FAISS_INDEX_PATH_BIO,
    FAISS_MAPPING_PATH_BIO,
    FAISS_INDEX_PATH_HCC,
    FAISS_MAPPING_PATH_HCC,
    # 数据路径配置
    INPUT_DIR,
    FACT_JSONL,
    FACT_JSONL_HCC,
    MESH_MAPPING_PATH,
    MESH_INFO_PATH,
    # 通用参数
    PROJECT,
    BATCH_SIZE,
)


def set_argument():
    """
    向后兼容旧的 config.set_argument() 接口。
    优先级：命令行参数 > 环境变量 > 默认值。
    """
    from . import settings

    parser = argparse.ArgumentParser(prog="BELIEF", description="BELIEF Pipeline Configuration")
    parser.add_argument("--project", type=str, default=settings.PROJECT)
    parser.add_argument("--neo4j_usr", type=str, default=settings.NEO4J_USR)
    parser.add_argument("--neo4j_pwd", type=str, default=settings.NEO4J_PWD)
    parser.add_argument("--neo4j_url", type=str, default=settings.NEO4J_URL)
    parser.add_argument("--mongo_url", type=str, default=settings.MONGO_URL)
    parser.add_argument("--db_name", type=str, default=settings.DB_NAME)
    parser.add_argument("--collection_name", type=str, default=settings.COLLECTION_NAME)
    parser.add_argument("--api_url_gpt", type=str, default=settings.API_URL)
    parser.add_argument("--api_key_gpt", type=str, default=settings.API_KEY)
    parser.add_argument("--fact_jsonl", type=str, default=settings.FACT_JSONL)
    parser.add_argument("--embedding_model_zh", type=str, default=settings.EMBEDDING_MODEL_ZH)
    parser.add_argument("--faiss_index_path_omic", type=str, default=settings.FAISS_INDEX_PATH_OMIC)
    parser.add_argument("--faiss_mapping_path_omic", type=str, default=settings.FAISS_MAPPING_PATH_OMIC)
    parser.add_argument("--embedding_model_en", type=str, default=settings.EMBEDDING_MODEL_EN)
    parser.add_argument("--faiss_index_path_bio", type=str, default=settings.FAISS_INDEX_PATH_BIO)
    parser.add_argument("--faiss_mapping_path_bio", type=str, default=settings.FAISS_MAPPING_PATH_BIO)
    parser.add_argument("--input_dir", type=str, default=settings.INPUT_DIR)
    parser.add_argument("--fact_jsonl_hcc", type=str, default=settings.FACT_JSONL_HCC)
    parser.add_argument("--faiss_index_path_hcc", type=str, default=settings.FAISS_INDEX_PATH_HCC)
    parser.add_argument("--faiss_mapping_path_hcc", type=str, default=settings.FAISS_MAPPING_PATH_HCC)
    parser.add_argument("--mesh_mapping_path", type=str, default=settings.MESH_MAPPING_PATH)
    parser.add_argument("--mesh_info_path", type=str, default=settings.MESH_INFO_PATH)
    parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE)
    return parser.parse_args()
