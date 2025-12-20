# 文献检索模块
from config import set_argument
from faiss_util.bio_faiss import BioPubmedFaissUtils
from faiss_util.hcc_faiss import HccPubmedFaissUtils
args = set_argument()

class hcc_pubmed_data:
    def __init__(self):
        self.args = args
    def build_hcc_faiss(self):
        builder = HccPubmedFaissUtils(self.args)
        builder.build_faiss()

    def search_hcc_faiss(self, query, top_k=3):
        builder = HccPubmedFaissUtils(self.args)
        res = builder.search(query, top_k)
        return res

class bio_pubmed_data:
    def __init__(self):
        self.args = args
        # 优化：模型只加载一次
        self.builder = BioPubmedFaissUtils(self.args)

    def build_bio_faiss(self):
        # 记得重新运行 build 来生效新的索引结构
        self.builder.build_index()

    def search_bio_faiss(self, vector_query, original_query, top_k=5):
        """
        vector_query: 用于召回 (Query + MeSH)
        original_query: 用于重排 (Query)
        """
        # 注意 fetch_k 设大一点 (50)，给重排留空间
        res = self.builder.search(vector_query, original_query, top_k=top_k, fetch_k=50)
        return res