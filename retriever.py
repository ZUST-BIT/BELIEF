# 整体检索模块
from intent_router import IntentRouter
from ne4j_retriever import GraphRAGRetriever
from omic import OmicData
from pubmed import hcc_pubmed_data, bio_pubmed_data
from util.data_refiner import EvidenceRefiner
from mesh_util import MeshManager
def retrieve_process(current_search_query):
    # 1. 初始化模块
    intent_router = IntentRouter()
    ne4j_retriever = GraphRAGRetriever()
    # omic_data_source = OmicData()
    # hcc_pubmed = hcc_pubmed_data()
    mesh_manager = MeshManager()
    bio_pubmed = bio_pubmed_data()
    refiner = EvidenceRefiner()

    # 2. 意图识别 & 实体提取
    query_intent = intent_router.intent_router(current_search_query)
    raw_entities = query_intent.get('extracted_entities', [])
    rewriten_query = query_intent.get('rewritten_query', current_search_query)
    # 3. MeSH 标准化 & 扩展
    mesh_results = mesh_manager.normalize(raw_entities)
    neo4j_search_set = set() # 用集合自动去重，用于查图谱
    expansion_terms = []     # 用于查文献的扩展词列表
    mesh_knowledge_text = "" # 用于给大模型看的定义

    if mesh_results:
        for orig_name, info in mesh_results.items():
            # A. 始终保留原始词
            neo4j_search_set.add(orig_name)
            
            if info['found']:
                std_name = info['standard_name']
                mesh_id = info['mesh_id']
                desc = info['description']
                
                # B. 加入 MeSH 标准名
                neo4j_search_set.add(std_name)
                expansion_terms.append(std_name)
                
                # C. 处理倒置名 (例如: 'Hearing Loss, Sensorineural' -> 'Sensorineural Hearing Loss')
                if ',' in std_name:
                    parts = [p.strip() for p in std_name.split(',')]
                    # 通常 MeSH 是两部分倒置
                    if len(parts) == 2:
                        reversed_name = f"{parts[1]} {parts[0]}"
                        neo4j_search_set.add(reversed_name)
                        expansion_terms.append(reversed_name)
                        print(f"     🔀 Rev: '{reversed_name}' (Natural Order)")

                # D. 收集描述作为知识背景
                if desc:
                    # 避免重复添加相同的描述
                    if desc not in mesh_knowledge_text:
                        mesh_knowledge_text += f"MeSH Term: {std_name}\nDescription: {desc}\n\n"
            else:
                pass
    else:
        # 如果没有 MeSH 结果，回退到原始实体
        neo4j_search_set = set(raw_entities)
    # print(f"重写实体：{raw_entities}")
    # 将集合转回列表，供检索器使用
    final_search_list = list(neo4j_search_set)
    # 3. 构造扩展后的向量查询 (Expanded Query)
    expansion_str = " ".join(set(expansion_terms))
    vector_search_query = f"{rewriten_query} {expansion_str}".strip()
    # A. 知识图谱检索 (Neo4j)
    neo4j_data = ne4j_retriever.retrieve(raw_entities)
    
    # B. 文献检索 (PubMed)
    # vector_query -> 用于 FAISS 粗排 (高召回)
    # original_query -> 用于 Cross-Encoder 精排 (高精度)
    bio_data = bio_pubmed.search_bio_faiss(
        vector_query=vector_search_query,
        original_query=current_search_query
    )
    # omic_resutls = omic_data_source.search_omic_data(current_search_query)
    # hcc_data = hcc_pubmed.search_hcc_faiss(current_search_query)
    # 4. 数据清洗与整合
    refined_data = refiner.run(
        kg_data=neo4j_data, 
        # omic_data=omic_resutls, 
        # hcc_data=hcc_data, 
        bio_data=bio_data
    )
    
    # 5. 将 MeSH 定义拼接到最前面
    if mesh_knowledge_text:
        refined_data = f"=== 📚 MeSH Term Definitions ===\n{mesh_knowledge_text}\n{refined_data}"

    return refined_data