# PubMed 文献读取、构建与检索工具
import os
import json
import faiss
import torch
import numpy as np
from tqdm import tqdm
from pymongo import MongoClient
from bson.objectid import ObjectId
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder 

class BioPubmedFaissUtils:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index_path = args.faiss_index_path_bio
        self.mapping_path = args.faiss_mapping_path_bio
        
        # --- 1. 加载 Embedding 模型 ---
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_en)
            self.model = AutoModel.from_pretrained(args.embedding_model_en)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"❌ Embedding 模型加载失败: {e}")
            exit(1)

        # --- 2. 加载 Reranker 模型 ---
        rerank_model_name = getattr(args, 'rerank_model', "cross-encoder/ms-marco-MiniLM-L-6-v2")
        try:
            self.reranker = CrossEncoder(rerank_model_name, device=self.device)
        except Exception as e:
            print(f"⚠️ Reranker 模型加载失败: {e}")
            self.reranker = None

        # --- 3. 初始化 Mongo 连接 ---
        try:
            self.client = MongoClient(self.args.mongo_url)
            self.collection = self.client[self.args.db_name][self.args.collection_name]
        except Exception as e:
            print(f"❌ MongoDB 连接失败: {e}")
            exit(1)

        self.index = None
        self.id_map = None

    @torch.no_grad()
    def embed_batch(self, texts):
        if not texts: return np.array([])
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def build_index(self):
        """ 构建索引：加入 MeSH 和 Keyword 语义增强 """
        total_docs = self.collection.count_documents({})
        if total_docs == 0: return

        all_embeddings = []
        id_mapping = {}
        
        # 1. 多读两个字段: concept, keyword
        cursor = self.collection.find({}, {"title": 1, "abstract": 1, "concept": 1, "keyword": 1, "_id": 1})
        
        batch_texts = []
        batch_mongo_ids = []
        global_faiss_id = 0 
        pbar = tqdm(total=total_docs, desc="Building Enriched Index")
        
        for doc in cursor:
            title = doc.get('title', '') or ''
            abstract = doc.get('abstract', '') or ''
            
            # 2. 提取 MeSH 词 (concept 字段是 list of dict)
            concepts = doc.get('concept', [])
            # 提取非空的 name
            mesh_names = [c.get('name') for c in concepts if isinstance(c, dict) and c.get('name')]
            mesh_str = ", ".join(mesh_names[:10]) # 取前10个重要概念
            
            # 3. 提取 Keywords (keyword 字段是 list of str)
            keywords = doc.get('keyword', [])
            kw_str = ", ".join(keywords[:5] if keywords else [])

            # 4. 拼接富文本：Title + Abstract + [MeSH] + [Keywords]
            text = f"{title.strip()}. {abstract.strip()}"
            if mesh_str:
                text += f" [MeSH: {mesh_str}]"
            if kw_str:
                text += f" [Keywords: {kw_str}]"
            
            if len(text) < 5: 
                pbar.update(1)
                continue

            batch_texts.append(text)
            batch_mongo_ids.append(str(doc["_id"]))

            if len(batch_texts) >= self.args.batch_size:
                embs = self.embed_batch(batch_texts)
                all_embeddings.append(embs)
                for m_id in batch_mongo_ids:
                    id_mapping[global_faiss_id] = m_id
                    global_faiss_id += 1
                batch_texts = []
                batch_mongo_ids = []
                pbar.update(self.args.batch_size)

        if batch_texts:
            embs = self.embed_batch(batch_texts)
            all_embeddings.append(embs)
            for m_id in batch_mongo_ids:
                id_mapping[global_faiss_id] = m_id
                global_faiss_id += 1
            pbar.update(len(batch_texts))
        pbar.close()

        if not all_embeddings: return
        final_embeddings = np.vstack(all_embeddings).astype("float32")
        dim = final_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim) 
        index.add(final_embeddings)
        faiss.write_index(index, self.index_path)
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(id_mapping, f)

    def _load_resources(self):
        if self.index is None:
            if not os.path.exists(self.index_path) or not os.path.exists(self.mapping_path):
                print(f"❌ 索引不存在")
                return False
            self.index = faiss.read_index(self.index_path)
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
        return True

    def search(self, vector_query, original_query, top_k=3, fetch_k=50):
        """
        :param vector_query: 扩展后的 Query (包含 MeSH 词)，用于 FAISS 召回
        :param original_query: 原始自然语言 Query，用于 Cross-Encoder 重排
        """
        if not vector_query: return []
        if not self._load_resources(): return []

        # 1. 向量检索 (使用 Expanded Query)
        q_vec = self.embed_batch([vector_query]) 
        scores, ids = self.index.search(q_vec, fetch_k)

        # 2. 获取 MongoDB 文档
        mongo_ids = []
        faiss_id_to_score = {}
        
        for score, idx in zip(scores[0], ids[0]):
            if str(idx) in self.id_map:
                m_id = self.id_map[str(idx)]
                mongo_ids.append(ObjectId(m_id)) 
                faiss_id_to_score[m_id] = float(score)

        cursor = self.collection.find({"_id": {"$in": mongo_ids}})
        docs_map = {str(d["_id"]): d for d in cursor}

        # 3. 准备重排数据
        candidate_pairs = []
        candidate_docs = []
        
        for m_id_obj in mongo_ids:
            m_id_str = str(m_id_obj)
            doc = docs_map.get(m_id_str)
            if doc:
                title = doc.get('title', 'No Title')
                abstract = doc.get('abstract', '')
                
                # 构造文档内容 (用于 LLM 阅读)
                content_for_llm = f"Title: {title}\nAbstract: {abstract}"
                
                # 构造文档内容 (用于 Rerank) - 加上 MeSH 增加匹配度
                concepts = [c.get('name') for c in doc.get('concept', []) if isinstance(c, dict) and c.get('name')]
                concept_str = ", ".join(concepts[:5])
                content_for_rerank = f"{title}. {abstract}. Keywords: {concept_str}"

                # --- 关键策略：重排使用原始问题 vs 增强文档 ---
                candidate_pairs.append([original_query, content_for_rerank])
                
                item = {
                    "id": m_id_str,
                    "vector_score": faiss_id_to_score.get(m_id_str, 0.0),
                    "title": title,
                    "abstract": abstract,
                    "content": content_for_llm
                }
                candidate_docs.append(item)

        # 4. 执行重排
        if self.reranker and candidate_pairs:
            rerank_scores = self.reranker.predict(candidate_pairs)
            for i, score in enumerate(rerank_scores):
                candidate_docs[i]['rerank_score'] = float(score)
            candidate_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        else:
            candidate_docs.sort(key=lambda x: x['vector_score'], reverse=True)

        return candidate_docs[:top_k]