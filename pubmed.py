"""Local PubMed FAISS index wrappers."""

from config import set_argument
from faiss_util.bio_faiss import get_bio_faiss_instance
from faiss_util.hcc_faiss import HccPubmedFaissUtils


args = set_argument()


class HccPubmedData:
    def __init__(self):
        self.args = args

    def build_hcc_faiss(self):
        builder = HccPubmedFaissUtils(self.args)
        builder.build_faiss()

    def search_hcc_faiss(self, query, top_k=3):
        builder = HccPubmedFaissUtils(self.args)
        return builder.search(query, top_k)


class BioPubmedData:
    def __init__(self):
        self.args = args
        self.builder = get_bio_faiss_instance(self.args)

    def build_bio_faiss(self):
        self.builder.build_index()

    def search_bio_faiss(self, vector_query, original_query, top_k=3):
        return self.builder.search(
            vector_query,
            original_query,
            top_k=top_k,
            fetch_k=50,
        )


# Backward-compatible aliases for older imports.
hcc_pubmed_data = HccPubmedData
bio_pubmed_data = BioPubmedData
