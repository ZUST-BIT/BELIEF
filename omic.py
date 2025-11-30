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


# if __name__ == '__main__':
#     omic = OmicData()
#     omic.build_omic_faiss()
#     res = omic.search_omic_data('针对Lenvatinib的药物敏感性来说，结合组学的信息，哪类特征（包括基因突变，包括临床特征）的病人更容易出现对Lenvatinib的耐药现象，这类病人在基因组的突变或者转录组的差异基因上有什么特征。')
#     print(res)
#     for i in res:
#         print("====文档====")
#         print("相似度:",i['score'])
#         print("ID:",i['docid'])
#         print("文档内容:",i['text'])