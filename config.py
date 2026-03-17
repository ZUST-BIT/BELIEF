# 系统配置参数
import argparse

# ==================== Think 模式控制 ====================
# 针对 Qwen3、QwQ 等带推理思考模式的模型：
#   True  - 通过 extra_body 关闭 think 模式，节省 token，适合结构化 JSON 输出任务
#   False - 保持模型默认行为（不发送 enable_thinking 参数）
# 注意：仅对支持 enable_thinking 参数的 API 端点生效（如 vLLM + Qwen3）
#       如果 API 端点不支持该参数，设置 True 可能导致请求报错，建议先测试
DISABLE_THINKING = True
# ========================================================

def set_argument():
    parser = argparse.ArgumentParser(prog='MEDAR-QA', description='Medical Question Answering System')
    parser.add_argument('--project',type=str,default='MEDAR-QA')
    parser.add_argument('--neo4j_usr',type=str,default='neo4j')
    # parser.add_argument('--neo4j_pwd',type=str,default='12345678')
    parser.add_argument('--neo4j_pwd',type=str,default='bitlab512')
    # parser.add_argument('--neo4j_url',type=str,default='bolt://localhost:7687')
    parser.add_argument('--neo4j_url',type=str,default='bolt://172.18.51.200:7687')
    # parser.add_argument('--mongo_url',type=str,default='mongodb://172.18.51.200:27017/')
    parser.add_argument('--mongo_url',type=str,default='mongodb://localhost:27017/')
    parser.add_argument('--db_name',type=str,default='bio')
    parser.add_argument('--collection_name',type=str,default='pubmed')
    # parser.add_argument('--api_url',type=str,default='https://api.deepseek.com')
    # parser.add_argument('--api_key',type=str,default='sk-0f17b61caf3f48e99944865634bd3a1c')
    parser.add_argument('--api_url_gpt',type=str,default='https://api.gptsapi.net/v1')
    parser.add_argument('--api_key_gpt',type=str,default='sk-IGQ8241037bdbeccfb18105d4774dc98ac20067097dQ3dDL')
    parser.add_argument('--fact_jsonl',type=str,default='datafile/fact_corpus_explanatory.jsonl')
    parser.add_argument('--embedding_model_zh',type=str,default='BAAI/bge-large-zh')
    parser.add_argument('--faiss_index_path_omic',type=str,default='output/faiss_index_omic.bin')
    parser.add_argument('--faiss_mapping_path_omic',type=str,default='output/faiss_mapping_omic.json')
    parser.add_argument('--embedding_model_en',type=str,default='BAAI/bge-large-en-v1.5')
    parser.add_argument('--faiss_index_path_bio',type=str,default='output/faiss_index_bio.bin')
    parser.add_argument('--faiss_mapping_path_bio',type=str,default='output/faiss_mapping_bio.json')
    parser.add_argument('--input_dir',type=str,default='D:/BitLabData/bio关键文档/outputs_final')
    parser.add_argument('--fact_jsonl_hcc',type=str,default='D:/BitLabData/bio关键文档/facts_cache.jsonl')
    parser.add_argument('--faiss_index_path_hcc',type=str,default='output/faiss_index_hcc.bin')
    parser.add_argument('--faiss_mapping_path_hcc',type=str,default='output/faiss_mapping_hcc.json')
    parser.add_argument('--mesh_mapping_path', type=str, default="D:/BitLabData/MeSH/mesh_mapping.json")
    parser.add_argument('--mesh_info_path', type=str, default="D:/BitLabData/MeSH/mesh_info.json")
    parser.add_argument('--batch_size',type=int,default=32)
    args = parser.parse_args()
    return args