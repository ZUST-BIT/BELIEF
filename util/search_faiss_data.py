""" 将用户输入的查询转换为向量，并使用 FAISS 索引进行搜索 """
import os
import pickle
from tqdm import tqdm

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
from util.build_faiss_data import process_pubmed_data
# MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# MODEL_NAME = 'BAAI/bge-large-en-v1.5'
MODEL_NAME = 'intfloat/multilingual-e5-base'
OUT_DIR = './pubmed_data'

EMBDDINGS_NPY = os.path.join(OUT_DIR, 'embeddings.npy')
ID_MAP_PKL = os.path.join(OUT_DIR, 'id_map.pkl')
DOCS_META_PKL = os.path.join(OUT_DIR, 'docs_meta.pkl')
FAISS_INDEX_FILE = os.path.join(OUT_DIR, 'paper.index')

class Searcher:
    def __init__(self):
        print("Initializing Searcher...")
        self.model = SentenceTransformer(MODEL_NAME)
        if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(ID_MAP_PKL) or not os.path.exists(DOCS_META_PKL):
            print("开始构建索引...")
            process_pubmed_data()
        self.index = faiss.read_index(FAISS_INDEX_FILE)
        # 加载已有的 ID 映射和元数据
        with open(ID_MAP_PKL, 'rb') as f:
            self.id_map = pickle.load(f)
        with open(DOCS_META_PKL, 'rb') as f:
            docs_meta = pickle.load(f)
        self.docs_meta_map = {meta['id']: meta for meta in docs_meta}
        print("Searcher initialized.")

    def search_pubmed_data(self, query,k):
        query_text = f"query:{query}"
        query_embedding = self.model.encode(
            [query_text],
            convert_to_tensor=True,
            show_progress_bar=False
        )
        query_embedding = query_embedding.cpu().numpy().astype('float32')

        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            doc_id = self.id_map[index]
            meta = self.docs_meta_map.get(doc_id)
            if meta:
                results.append({
                    "score":float(score),
                    "title":meta.get("title"),
                    "abstract":meta.get("abstract"),
                    "doi":meta.get("doi"),
                    "id":doc_id
                })
        return results


model = SentenceTransformer("paraphrase-Multilingual-MiniLM-L12-v2")
def search_omics_data(query, top_k):
    """ 处理用户问题，返回最相似的 top_k 条记录 """
    jsonl_file = './omic_data/table_rag_entries.jsonl'
    faiss_index_file = './omic_data/faiss.index'
    embedding_file = './omic_data/embeddings.npy'
    if os.path.exists(faiss_index_file) and os.path.exists(embedding_file):
        index = faiss.read_index(faiss_index_file)
        embeddings = np.load(embedding_file)
    else:
        print("开始构建索引...")
        from util.build_faiss_data import process_omics_data
        index, embeddings = process_omics_data()
    query_emd = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    scores, indices = index.search(query_emd, top_k)
    results = []
    records = []
    # 读取 JSONL 文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            records.append(data)
    for score, idx in zip(scores[0], indices[0]):
        record = records[idx]
        results.append({
            "score": float(score),
            "id": record["id"],
            "text": record["text"][:500],
        })
    return results