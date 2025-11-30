import os
import json
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class HccPubmedFaissUtils:
    def __init__(self, args):
        self.args = args

        # print(f"[INFO] 正在加载模型: {args.embedding_model_en}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_en)
        self.model = AutoModel.from_pretrained(args.embedding_model_en)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"[INFO] 运行设备: {self.device}")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, text):
        """
        生成向量
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)
        # 取 CLS token (第一个 token) 的向量作为句子表示
        vec = outputs.last_hidden_state[:, 0, :]
        # L2 正则化 (对于余弦相似度检索非常重要)
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec.cpu().numpy()[0]

    def preprocess_grobid_files(self):
        """
        读取文件夹里所有的 JSON，处理成标准的文档列表 (docs)
        同时保存一个 cache.jsonl 方便后续检索时读取原文
        """
        print(f"[INFO] 正在读取 Grobid JSON 目录: {self.args.input_dir}")
        docs = []
        files = [f for f in os.listdir(self.args.input_dir) if f.endswith(".json")]
        
        doc_counter = 0

        for filename in tqdm(files, desc="处理文件"):
            file_path = os.path.join(self.args.input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 1. 获取基本信息
            article_title = data.get("title", "Unknown Title")
            
            # 2. 处理摘要
            abstract = data.get("abstract", "")
            if abstract and len(abstract) > 10:
                doc_id = f"{filename}_abstract"
                # 拼接文本用于向量化
                text_to_embed = f"Paper: {article_title}\nSection: Abstract\nContent: {abstract}"
                
                docs.append({
                    "id": doc_id,
                    "text": text_to_embed, # 用于 embedding
                    "metadata": {          # 用于展示
                        "source": filename,
                        "title": article_title,
                        "section": "Abstract",
                        "content": abstract
                    }
                })
                doc_counter += 1

            # 3. 处理正文章节
            body = data.get("body", [])
            for idx, section in enumerate(body):
                sec_title = section.get("section_title", "No Title")
                paragraphs = section.get("content", [])
                sec_content = "\n".join(paragraphs)
                
                if len(sec_content) < 5: continue

                doc_id = f"{filename}_sec_{idx}"
                # 核心拼接逻辑
                text_to_embed = f"Paper: {article_title}\nSection: {sec_title}\nContent: {sec_content}"

                docs.append({
                    "id": doc_id,
                    "text": text_to_embed,
                    "metadata": {
                        "source": filename,
                        "title": article_title,
                        "section": sec_title,
                        "content": sec_content
                    }
                })
                doc_counter += 1
        
        # 保存处理好的中间文件 (JSONL格式)，供 build_faiss 和 search 使用
        print(f"[INFO] 保存预处理数据到: {self.args.fact_jsonl_hcc}")
        with open(self.args.fact_jsonl_hcc, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        return docs

    def build_faiss(self):
        # 1. 先把散乱的 JSON 文件转成列表
        if os.path.exists(self.args.fact_jsonl_hcc):
            print(f"[INFO] 发现缓存文件，直接加载: {self.args.fact_jsonl_hcc}")
            docs = []
            with open(self.args.fact_jsonl_hcc, "r", encoding="utf-8") as f:
                for line in f:
                    docs.append(json.loads(line))
        else:
            docs = self.preprocess_grobid_files()

        print(f"[INFO] 总文档切片数量: {len(docs)}")

        embeddings = []
        id_map = {} # 映射: FAISS索引 ID (int) -> 真实文档 ID (str)

        print("[INFO] 开始 embedding 文本……")
        # 逐个生成向量
        for idx, doc in enumerate(tqdm(docs, desc="Embedding")):
            vec = self.embed(doc["text"])
            embeddings.append(vec)
            id_map[idx] = doc["id"]

        # 转换为 numpy 数组
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        print(f"[INFO] embedding 维度 = {dim}")

        # 构建 FAISS 索引
        print("[INFO] 构建 FAISS IndexFlatIP")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # 保存索引
        faiss.write_index(index, self.args.faiss_index_path_hcc)
        print(f"[SAVE] 向量索引保存到 {self.args.faiss_index_path_hcc}")

        # 保存 ID 映射
        with open(self.args.faiss_mapping_path_hcc, "w", encoding="utf-8") as f:
            json.dump(id_map, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] 文档映射保存到 {self.args.faiss_mapping_path_hcc}")

        print("[OK] FAISS 构建完成")

    def search(self, query, top_k=3):
        """
        检索函数 (已修复 NoneType 报错)
        """
        if not os.path.exists(self.args.faiss_index_path_hcc):
            print("❌ 索引文件不存在，请先运行 build_faiss()")
            return []

        # 1. 加载资源
        # print("[INFO] 正在加载 FAISS 索引...")
        index = faiss.read_index(self.args.faiss_index_path_hcc)
        
        # print("[INFO] 正在加载 ID 映射...")
        with open(self.args.faiss_mapping_path_hcc, "r", encoding="utf-8") as f:
            id_map = json.load(f)
        
        # 2. 加载原文数据 (建立 ID -> 文档 的内存索引)
        # print("[INFO] 正在加载原文缓存...")
        docs_dict = {}
        try:
            with open(self.args.fact_jsonl_hcc, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        docs_dict[d['id']] = d
                    except:
                        continue # 跳过损坏的行
        except FileNotFoundError:
            print(f"⚠️ 警告: 找不到原文缓存文件 {self.args.fact_jsonl_hcc}，检索结果将无法显示具体内容。")

        # 3. 编码查询语句
        q_vec = self.embed(query)
        q_vec = np.expand_dims(q_vec, axis=0) 

        # 4. 搜索
        scores, ids = index.search(q_vec, top_k)

        # 5. 格式化结果
        results = []
        
        for score, idx in zip(scores[0], ids[0]):
            real_doc_id = id_map.get(str(idx)) 
            
            if not real_doc_id:
                print(f"⚠️ 警告: 索引 ID {idx} 在映射文件中找不到对应的文档 ID")
                continue

            doc_info = docs_dict.get(real_doc_id, {})
            metadata = doc_info.get('metadata', {})

            title = metadata.get('title', 'Unknown Title')
            section = metadata.get('section', 'Unknown Section')
            content = metadata.get('content', '') # 默认为空字符串
            
            if content is None: content = ""

            results.append({
                "score": float(score),
                "docid": real_doc_id,
                "title": title,
                "section": section,
                "content": content
            })
        return results

# ================= 运行逻辑 =================
# if __name__ == "__main__":
#     # 1. 初始化配置
#     args = set_argument()
    
#     # 2. 初始化构建器
#     builder = HccPubmedFaissUtils(args)
    
#     # 3. 构建索引 (第一次运行或数据更新时运行)
#     # 如果你已经跑过一次，可以注释掉这行，直接跑 search
#     # builder.build_faiss()

#     # 4. 测试检索
#     test_query = "Regarding Lenvatinib drug sensitivity, combining omics information, what features (including gene mutations and clinical features) make patients more prone to Lenvatinib resistance?"

#     builder.search(test_query, top_k=3)