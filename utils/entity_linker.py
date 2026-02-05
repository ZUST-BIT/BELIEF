# 文件路径: utils/entity_linker.py
import os
import json
import pickle
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# --- 配置文件路径 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "kg_index_output")

# ==================== 单例模式 ====================
_entity_retriever_instance = None

def get_entity_retriever_instance():
    """
    获取 EntityRetriever 单例实例
    避免重复加载 SapBERT 模型，提高运行效率
    """
    global _entity_retriever_instance
    if _entity_retriever_instance is None:
        print("🔄 [Entity Linker] 首次初始化，加载 SapBERT 模型...")
        _entity_retriever_instance = EntityRetriever()
        print("✅ [Entity Linker] 模型加载完成，后续调用将复用此实例")
    return _entity_retriever_instance
# ================================================

class EntityRetriever:
    def __init__(self):
        self.dict_path = os.path.join(OUTPUT_DIR, "keyword_mapping.json")
        self.meta_path = os.path.join(OUTPUT_DIR, "entity_metadata.pkl")
        self.index_path = os.path.join(OUTPUT_DIR, "primekg_mesh.index")
        self.map_path = os.path.join(OUTPUT_DIR, "faiss_id_mapping.pkl")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sapbert_model = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        
        # 加载资源
        self._load_resources()

    def _load_resources(self):
        # print("🔄 [Linker] Loading Knowledge Base artifacts...")
        if not os.path.exists(self.dict_path):
            print(f"⚠️ Warning: Dictionary not found at {self.dict_path}")
            self.keyword_map = {}
        else:
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                self.keyword_map = json.load(f)

        if not os.path.exists(self.meta_path):
             self.metadata = {}
        else:
            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f)

        # 加载向量检索组件
        self.index = None
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.map_path, 'rb') as f:
                self.faiss_id_map = pickle.load(f)
            
            # 只有当索引存在时才加载 BERT 模型
            # print(f"🔄 [Linker] Loading SapBERT model on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.sapbert_model)
            self.model = AutoModel.from_pretrained(self.sapbert_model).to(self.device)
            self.model.eval()

    def _get_embedding(self, text):
        inputs = self.tokenizer([text], padding=True, truncation=True, 
                              max_length=64, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embed = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        faiss.normalize_L2(embed)
        return embed

    def search_list(self, entity_list, vector_top_k=1, score_threshold=0.75):
        """
        核心对齐方法
        """
        results = []
        for query in entity_list:
            if not query: continue
            
            query_norm = query.lower().strip()
            res_obj = {
                "input_query": query, 
                "status": "not_found", 
                "data": None
            }

            # 1. 字典精确匹配
            if query_norm in self.keyword_map:
                neo4j_id = self.keyword_map[query_norm]
                res_obj.update({
                    "status": "found", 
                    "match_type": "exact", 
                    "score": 1.0, 
                    "data": self.metadata.get(neo4j_id)
                })
                results.append(res_obj)
                continue

            # 2. 向量模糊匹配
            if self.index:
                query_vec = self._get_embedding(query)
                scores, indices = self.index.search(query_vec.astype('float32'), vector_top_k)
                top_score = float(scores[0][0])
                top_idx = int(indices[0][0])
                
                if top_score >= score_threshold and top_idx != -1:
                    neo4j_id = self.faiss_id_map[top_idx]
                    res_obj.update({
                        "status": "found", 
                        "match_type": "vector", 
                        "score": round(top_score, 4), 
                        "data": self.metadata.get(neo4j_id)
                    })
                    
            results.append(res_obj)
        return results
    

# if __name__ == "__main__":
#     retriever = EntityRetriever()
#     entity_list =  [
#         "chronic fatigue",
#         "hypertension",
#         "diabetes",
#         "nodules",
#         "Caspase-9",
#         "CD15",
#         "Cyclin-dependent kinase 4",
#         "chromosome 18"
#         # "CD15",
#         # "Cyclin-dependent kinase 4",
#         # "Ras pathway transcription factors"
#     ]
#     res = retriever.search_list(entity_list)
#     print(res)