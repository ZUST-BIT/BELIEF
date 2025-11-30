# faiss_utils.py
import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

class OmicFaissUtils:
    def __init__(self, args):
        self.args = args

        # print(f"[INFO] 加载 BGE-Large-ZH 模型: {args.embedding_model_zh}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_zh)
        self.model = AutoModel.from_pretrained(args.embedding_model_zh)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)
        vec = outputs.last_hidden_state[:, 0, :]
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec.cpu().numpy()[0]

    def build_faiss(self):
        print(f"[INFO] 加载事实库 JSONL: {self.args.fact_jsonl}")

        docs = []
        with open(self.args.fact_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))

        print(f"[INFO] 文档数量: {len(docs)}")

        embeddings = []
        id_map = {}

        print("[INFO] 开始 embedding 文本……")
        for idx, doc in enumerate(tqdm(docs)):
            vec = self.embed(doc["text"])
            embeddings.append(vec)
            id_map[idx] = doc["id"]

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        print(f"[INFO] embedding 维度 = {dim}")

        print("[INFO] 构建 FAISS IndexFlatIP")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, self.args.faiss_index_path_omic)
        print(f"[SAVE] 向量索引保存到 {self.args.faiss_index_path_omic}")

        with open(self.args.faiss_mapping_path_omic, "w", encoding="utf-8") as f:
            json.dump(id_map, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] 文档映射保存到 {self.args.faiss_mapping_path_omic}")

        print("[OK] FAISS 构建完成")

    def search(self, query, top_k=3):
        index = faiss.read_index(self.args.faiss_index_path_omic)
        with open(self.args.faiss_mapping_path_omic, "r", encoding="utf-8") as f:
            id_map = json.load(f)
        docs = []
        with open(self.args.fact_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))
        q_vec = self.embed(query)
        q_vec = np.expand_dims(q_vec, axis=0)
        scores, ids = index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            docid = id_map[str(idx)]

            doc = next(d for d in docs if d['id'] == docid)
            results.append({
                "score": float(score),
                "docid": docid,
                "text": doc['text'],
                "metadata": doc.get('metadata', {})
            })

        return results
        
