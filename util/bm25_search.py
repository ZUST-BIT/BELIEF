""" 利用实体识别的结果 + BM25检索文献 """
import os
import pickle
from tqdm import tqdm
from pymongo import MongoClient
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

MONGO_URI = "mongodb://172.18.51.200:27017/"
DB_name = 'bio'
COLLECTION_name = 'lc_pubmed'

OUT_DIR = './bm25_data_hcc'
BM25_FILE = os.path.join(OUT_DIR, 'bm25_model.pkl')
TOKENIZED_CORPUS_FILE = os.path.join(OUT_DIR, 'tokenized_corpus.pkl')
DOCS_META_FILE = os.path.join(OUT_DIR, 'docs_meta.pkl')

os.makedirs(OUT_DIR, exist_ok=True)

def fetch_docs_from_mongo(uri,db_name,collection_name):
    """ 从mongo中获取数据 """
    client = MongoClient(uri)
    col = client[db_name][collection_name]
    cursor = col.find({},{"title":1,"abstract":1,"doi":1})
    docs = list(cursor)
    client.close()
    return docs

def process_text(text):
    """ 处理文本，去除停用词，分词 """
    if not text:
        return []
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')) # 加载停词器 
    filtered_tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return filtered_tokens

def build_bm25_index(docs):
    """ 构建BM25索引 """
    print("Building BM25 index...")
    corpus = []
    docs_meta = []
    for d in tqdm(docs,desc='Preprocessing documents'):
        title = d.get('title', '') or ''
        abstract = d.get('abstract', '') or ''
        combined = (title + " " + abstract).strip()
        tokens = process_text(combined)
        if not tokens:
            continue
        corpus.append(tokens)
        docs_meta.append({
            "_id":str(d.get('_id')),
            "title":title,
            "abstract":abstract,
            "doi":d.get('doi', '')
        })

    print(f"Total docs in corpus: {len(corpus)}")
    print("Building BM25 model...")
    bm25 = BM25Okapi(corpus)
    return bm25,corpus,docs_meta

def save_all(bm25,corpus,docs_meta):
    """ 保存所有数据 """
    with open(BM25_FILE, 'wb') as f:
        pickle.dump(bm25, f)
    with open(TOKENIZED_CORPUS_FILE, 'wb') as f:
        pickle.dump(corpus, f)
    with open(DOCS_META_FILE, 'wb') as f:
        pickle.dump(docs_meta, f)
    print("BM25 index and data saved to",OUT_DIR)

def load_all():
    """ 如果文件存在，加载数据 """
    if os.path.exists(BM25_FILE) and os.path.exists(TOKENIZED_CORPUS_FILE) and os.path.exists(DOCS_META_FILE):
        print("Loading BM25 index...")
        with open(BM25_FILE, 'rb') as f:
            bm25 = pickle.load(f)
        with open(TOKENIZED_CORPUS_FILE, 'rb') as f:
            corpus = pickle.load(f)
        with open(DOCS_META_FILE, 'rb') as f:
            docs_meta = pickle.load(f)
        return bm25,corpus,docs_meta
    return None,None,None

def entities_to_tokens(res_dict,weight_map=None):
    """ 将实体字典转换为BM25检索的token列表 """
    if weight_map is None:
        weight_map = {'GENE':1.2,'DISEASE':1.1,'DRUG':1.0,'OTHER':0.5}

    tokens = []
    for ent in res_dict.get("entities",[]):
        etype = ent.get('type','OTHER')
        weight = weight_map.get(etype,1.0)
        names = [ent.get('standard_name','')] + ent.get('expanded_names',[])
        for n in names:
            tks = process_text(n)
            tokens.extend(tks * int(weight * 5))
    return tokens

def entity_based_bm25_search(bm25,corpus,docs_meta,res_dict,top_k=10):
    """ 基于实体的BM25检索 """
    tokens = entities_to_tokens(res_dict)
    if not tokens:
        print("No entity found in the result.")
        return []
    scores = bm25.get_scores(tokens)
    top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank,idx in enumerate(top_ids,1):
        results.append({
            "rank":rank,
            "score":float(scores[idx]),
            "title":docs_meta[idx]["title"],
            "abstract":docs_meta[idx]["abstract"],
            "doi":docs_meta[idx].get("doi", ""),
            "_id":docs_meta[idx]["_id"]
        })
    return results

def main(force_build=False):
    """ 主函数 """
    nltk.download('punkt',quiet=True)
    nltk.download('stopwords',quiet=True)
    if not force_build:
        bm25,corpus,docs_meta = load_all()
        if bm25 is not None:
            print("BM25 index and data loaded from",OUT_DIR)
            return bm25,corpus,docs_meta
    print("Fetching documents from MongoDB...")
    docs = fetch_docs_from_mongo(MONGO_URI,DB_name,COLLECTION_name)
    print(f"Total docs fetched: {len(docs)}")
    bm25,corpus,docs_meta = build_bm25_index(docs)
    save_all(bm25,corpus,docs_meta)
    return bm25,corpus,docs_meta



