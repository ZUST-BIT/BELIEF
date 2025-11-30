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
    def build_bio_faiss(self):
        builder = BioPubmedFaissUtils(self.args)
        builder.build_index()
    def search_bio_faiss(self, query, top_k=3):
        builder = BioPubmedFaissUtils(self.args)
        res = builder.search(query, top_k)
        return res
