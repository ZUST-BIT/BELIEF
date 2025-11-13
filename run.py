from util.extract_entity import extract_entity
from util.get_neo4j_data import query_nodes_info
from util.search_faiss_data import search_omics_data, Searcher
from util.bm25_search import main, entity_based_bm25_search
from util.answer import answer
query = '结合组学信息，不同的药物Lenvatinib、sorafenib、regorafenib、apatinib的耐药相关的基因是否存在共性，这些共性基因都是参与什么具体的肿瘤生物学途径的。'
entity_list = extract_entity(query)
graph_data = query_nodes_info(entity_list)
omics_data = search_omics_data(query, 10)
searcher = Searcher()
pubmed_data = searcher.search_pubmed_data(query, 10)
# print(pubmed_data)
bm25,corpus,docs_meta = main()
pubmed_data_bm25 = entity_based_bm25_search(bm25,corpus,docs_meta,entity_list)
final_res = answer(graph_data,pubmed_data,pubmed_data_bm25,omics_data,query)
print(final_res)