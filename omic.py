from config import set_argument
from faiss_util.omic_faiss import OmicFaissUtils
args = set_argument()
class OmicData:
    def __init__(self):
        self.args = args
    def build_omic_faiss(self):
        builder = OmicFaissUtils(self.args)
        builder.build_faiss()
    
    def search_omic_data(self, query, top_k=3):
        builder = OmicFaissUtils(self.args)
        res = builder.search(query, top_k)
        return res