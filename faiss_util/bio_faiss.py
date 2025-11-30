import os
import json
import faiss
import torch
import numpy as np
from tqdm import tqdm
from pymongo import MongoClient
from bson.objectid import ObjectId
from transformers import AutoTokenizer, AutoModel
from config import set_argument

class BioPubmedFaissUtils:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index_path = args.faiss_index_path_bio
        self.mapping_path = args.faiss_mapping_path_bio
        # 2. 加载模型
        # print(f"[INFO] 正在加载模型: {args.embedding_model_en} (Device: {self.device})")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_en)
            self.model = AutoModel.from_pretrained(args.embedding_model_en)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            exit(1)

        # 初始化 Mongo 连接
        try:
            self.client = MongoClient(self.args.mongo_url)
            self.collection = self.client[self.args.db_name][self.args.collection_name]
            # 测试连接
            self.client.server_info()
            # print(f"[INFO] MongoDB 连接成功: {self.args.mongo_url}")
        except Exception as e:
            print(f"❌ MongoDB 连接失败: {e}")
            exit(1)

    @torch.no_grad()
    def embed_batch(self, texts):
        """ 
        批量向量化 (Batch Embedding)
        """
        if not texts: return np.array([])
        
        # BGE 模型建议: 检索 Query 需要加前缀，但 Document 不需要
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)
        # CLS Pooling
        embeddings = outputs.last_hidden_state[:, 0, :]
        # L2 Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def build_index(self):
        """
        构建索引主流程：流式读取 -> 向量化 -> 存索引 -> 存映射
        """
        # 1. 统计总数
        total_docs = self.collection.count_documents({})
        # print(f"[INFO] 数据库中共有 {total_docs} 条文档，准备构建索引...")
        
        if total_docs == 0:
            print("❌ 数据库为空，终止构建。")
            return

        # 2. 准备容器
        all_embeddings = []
        id_mapping = {} # FAISS_INT_ID -> MONGO_STR_ID
        
        # 3. 游标流式读取 (只读需要的字段，减小网络压力)
        cursor = self.collection.find({}, {"title": 1, "abstract": 1, "_id": 1})
        
        batch_texts = []
        batch_mongo_ids = []
        global_faiss_id = 0 

        # 4. 循环处理
        pbar = tqdm(total=total_docs, desc="Building")
        
        for doc in cursor:
            # 数据清洗与拼接
            title = doc.get('title', '') or ''
            abstract = doc.get('abstract', '') or ''
            text = f"{title.strip()}. {abstract.strip()}"
            
            # 跳过过短的数据
            if len(text) < 5: 
                pbar.update(1)
                continue

            batch_texts.append(text)
            batch_mongo_ids.append(str(doc["_id"]))

            # 攒够一个 Batch 就处理
            if len(batch_texts) >= self.args.batch_size:
                embs = self.embed_batch(batch_texts)
                all_embeddings.append(embs)
                
                # 记录 ID 映射
                for m_id in batch_mongo_ids:
                    id_mapping[global_faiss_id] = m_id
                    global_faiss_id += 1
                
                # 清空缓存
                batch_texts = []
                batch_mongo_ids = []
                pbar.update(self.args.batch_size)

        # 处理剩余尾巴
        if batch_texts:
            embs = self.embed_batch(batch_texts)
            all_embeddings.append(embs)
            for m_id in batch_mongo_ids:
                id_mapping[global_faiss_id] = m_id
                global_faiss_id += 1
            pbar.update(len(batch_texts))
            
        pbar.close()

        # 5. 合并向量 & 构建 FAISS
        if not all_embeddings:
            print("❌ 未生成任何向量。")
            return

        final_embeddings = np.vstack(all_embeddings).astype("float32")
        dim = final_embeddings.shape[1]
        
        # 使用 Inner Product (因为已经归一化了，所以等于 Cosine)
        index = faiss.IndexFlatIP(dim) 
        index.add(final_embeddings)

        faiss.write_index(index, self.index_path)
        
        # print(f"[SAVE] 保存 ID 映射到: {self.mapping_path}")
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(id_mapping, f)

        # print("✅ 构建完成！")

    def search(self, query, top_k=3):
        """
        检索主流程：加载索引 -> 搜索 TopK -> 回查 MongoDB
        返回: List[Dict] (包含文档的详细信息)
        """
        if not query:
            print("❌ 检索模式下必须提供 --query 参数")
            return []

        if not os.path.exists(self.index_path) or not os.path.exists(self.mapping_path):
            print(f"❌ 索引文件不存在，请先运行 --mode build")
            return []

        # 1. 加载资源 (只加载必要的)
        index = faiss.read_index(self.index_path)
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)

        # 2. 向量检索
        q_vec = self.embed_batch([query]) 
        scores, ids = index.search(q_vec, top_k)

        # 3. 回查 MongoDB
        mongo_ids = []
        hits = [] # 暂存 (score, mongo_id)
        
        for score, idx in zip(scores[0], ids[0]):
            if str(idx) in id_map:
                m_id = id_map[str(idx)]
                mongo_ids.append(ObjectId(m_id)) 
                hits.append({"score": score, "id": m_id})
        
        # 批量查询
        cursor = self.collection.find({"_id": {"$in": mongo_ids}})
        docs_dict = {str(d["_id"]): d for d in cursor}

        # 4. 组装返回结果
        results = []
        
        for hit in hits:
            doc = docs_dict.get(hit['id'])
            score = float(hit['score']) # 转为 float 以便 JSON 序列化
            
            if doc:
                # 提取需要的字段，并处理可能的空值
                item = {
                    "id": hit['id'],
                    "score": score,
                    "title": doc.get('title', 'No Title'),
                    "abstract": doc.get('abstract', ''),
                    "doi": doc.get('doi', ''),
                    "keyword": doc.get('keyword', []), # 保留关键词列表
                    # 拼接一个用于 RAG 的完整文本内容
                    "content": f"Title: {doc.get('title', '')}\nAbstract: {doc.get('abstract', '')}"
                }
                results.append(item)
            else:
                # 即使没找到文档，也建议记录下来方便 debug，或者直接跳过
                print(f"⚠️ Warning: ID {hit['id']} in index but not in MongoDB.")
                
        return results
# ================= 入口 =================
# if __name__ == "__main__":
#     args = set_argument()
    
#     system = BioPubmedFaissUtils(args)
#     query = "Regarding Lenvatinib drug sensitivity, combining omics information, what features (including gene mutations and clinical features) make patients more prone to Lenvatinib resistance?"
#     # system.build_index()
#     system.search(query, top_k=3)