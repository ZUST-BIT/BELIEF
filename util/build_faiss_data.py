""" 获取 mongodb 数据 然后向量化为句子向量 """
import os
import pickle
from tqdm import tqdm

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json

# MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# MODEL_NAME = 'BAAI/bge-m3'

def process_pubmed_data():
    """ 处理pubmed数据 """
    from util.get_mongodb_data import get_and_convert_data
    MODEL_NAME = 'intfloat/multilingual-e5-base'
    OUT_DIR = './vector_data'
    EMBDDINGS_NPY = os.path.join(OUT_DIR, 'embeddings.npy')
    ID_MAP_PKL = os.path.join(OUT_DIR, 'id_map.pkl')
    DOCS_META_PKL = os.path.join(OUT_DIR, 'docs_meta.pkl')
    FAISS_INDEX_FILE = os.path.join(OUT_DIR, 'paper.index')
    BATCH_SIZE = 64
    os.makedirs(OUT_DIR, exist_ok=True)

    texts, metas = get_and_convert_data()
    embeddings = encode_texts(texts, MODEL_NAME, BATCH_SIZE)
    # index = build_index(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    id_map = [meta['id'] for meta in metas]
    np.save(EMBDDINGS_NPY, embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(ID_MAP_PKL, 'wb') as f:
        pickle.dump(id_map, f)
    with open(DOCS_META_PKL, 'wb') as f:
        pickle.dump(metas, f)


def encode_texts(texts, model_name, batch_size) -> np.ndarray:
    """ 编码文本数据 """
    model = SentenceTransformer(model_name)
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size),desc='encoding'):
        # batch_texts = texts[i:i + batch_size]
        batch_texts = [f"passage:{t}" for t in texts[i:i + batch_size]]
        embs = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
        all_embs.append(embs)
    embs = np.vstack(all_embs).astype('float32')
    # L2 归一化
    faiss.normalize_L2(embs)
    return embs

