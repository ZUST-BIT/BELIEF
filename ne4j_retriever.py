# 检索知识图谱代码
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from config import set_argument
from neo4j import Neo4jManager

class GraphRAGRetriever:
    def __init__(self, index_file='output/faiss_index_neo4j.index', meta_file='output/entity_metadata.pkl'):
        """
        初始化检索器：加载 Embedding 模型、Faiss 索引和 Neo4j 连接。
        """
        args = set_argument()
        
        # 1. 加载 Embedding 模型
        self.model = SentenceTransformer(args.embedding_model_en)
        
        # 2. 加载 Faiss 索引和元数据
        self.index = faiss.read_index(index_file)
        with open(meta_file, 'rb') as f:
            self.metadata = pickle.load(f)
        self.graph_manager = Neo4jManager()
        

    def _vector_search(self, query_text, top_k=1, threshold=0.7):
        """内部方法：单个实体的向量匹配"""
        query_vector = self.model.encode([query_text])
        faiss.normalize_L2(query_vector)
        
        D, I = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        idx_id = I[0][0]
        score = D[0][0]
        
        if idx_id == -1 or score < threshold:
            return None, score
            
        return self.metadata[idx_id]['name'], score

    def retrieve(self, entity_list):
        """
        [主函数]
        输入: 实体列表 (e.g. ['cisplatin', 'lung cancer'])
        输出: 包含检索结果的字典
        """
        # Step 1: 实体链接 (Entity Linking)
        standard_entities = set()
        
        for entity in entity_list:
            match_name, score = self._vector_search(entity)
            if match_name:
                standard_entities.add(match_name)
            # else:
            #     print(f"  No match found for '{entity}'")
        
        standard_list = list(standard_entities)
        if not standard_list:
            return {}

        # Step 2: 图谱路径查询 (Graph Routing)
        retrieval_results = self.graph_manager.query_node_info(standard_list)
        
        return retrieval_results