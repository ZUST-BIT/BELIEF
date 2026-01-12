# retriever.py
import json
from intent_router import IntentRouter
from neo4j import Neo4jManager
from pubmed import bio_pubmed_data
from utils.data_refiner import EvidenceRefiner
from utils.entity_linker import EntityRetriever # 导入刚才写好的类

def retrieve_process(current_search_query):
    """
    基于 IntentRouter 的检索计划执行 RAG 流程 (Hybrid Search)
    """
    # 1. 初始化模块
    intent_router = IntentRouter()
    neo4j_manager = Neo4jManager()
    bio_pubmed = bio_pubmed_data()
    refiner = EvidenceRefiner()
    entity_linker = EntityRetriever()
    
    # 2. 获取检索计划 (Intent & Plan)
    plan_result = intent_router.intent_router(current_search_query)
    final_evidence_list = []
    
    entities_to_align = []
    
    # 提取问题中的实体 normalized 名称
    if "question_entities" in plan_result:
        for ent in plan_result["question_entities"]:
            # if ent.get("normalized"):
            entities_to_align.append(ent)

    # 去重
    entities_to_align = list(set(entities_to_align))
    # 1.2 执行混合检索对齐 (Dictionary + Vector)
    alignment_results = entity_linker.search_list(entities_to_align)
    # 1.3 处理对齐结果
    kg_id_list = []        # 用于查 Neo4j 的 ID
    expansion_terms = []   # 用于查 PubMed 的扩展词
    aligned_info_blocks = [] # 用于返回给 LLM 的实体信息块
    
    for res in alignment_results:
        if res["status"] == "found":
            data = res["data"]
            
            # A. 收集 ID 用于 KG 检索
            if "id" in data:
                kg_id_list.append(data["id"])
            
            # B. 收集名称用于 Query Expansion
            # 优先使用标准名
            std_name = data.get("std_name")
            if std_name:
                expansion_terms.append(std_name)
            
            # 补充别名 (限制前2个，防止 query 过长)
            aliases = data.get("aliases", [])
            if aliases:
                expansion_terms.extend(aliases[:2])
                
            # C. 构建实体信息块 (Evidence)
            info_text = (
                f"Entity: {data.get('ori_name')} (Std: {std_name})\n"
                f"Definition: {data.get('description')}\n"
                f"Type: {data.get('labels')}\n"
                f"MeSH ID: {data.get('mesh_id')}"
            )
            aligned_info_blocks.append(info_text)
            
    # 将对齐成功的实体信息作为第一条证据保存
    if aligned_info_blocks:
        final_evidence_list.append({
            "source_type": "Entity Definitions (Knowledge Base)",
            "content": "\n---\n".join(aligned_info_blocks),
            "metadata": {"source": "Local Hybrid Index"}
        })

    if kg_id_list:
        # 去重 ID
        kg_id_list = list(set(kg_id_list))
        neo4j_data = neo4j_manager.query_subgraph_by_ids(kg_id_list) 
        if neo4j_data:
            kg_content = refiner.format_kg_data(neo4j_data)
            if len(kg_content) > 10:
                final_evidence_list.append({
                    "source_type": "Knowledge Graph (PrimeKG)",
                    "content": kg_content,
                    "metadata": {"matched_ids": kg_id_list}
                })
    else:
        print("🕸️ [KG Search] No valid entity IDs found, skipping KG search.")


    base_query = plan_result.get("rewritten_query", current_search_query)
    
    # 简单去重并拼接扩展词
    unique_expansion = list(set(expansion_terms))
    filtered_expansion = [t for t in unique_expansion if t.lower() not in base_query.lower()]
    
    final_vector_query = f"{base_query} {' '.join(filtered_expansion[:5])}".strip()
    
    # 3.2 执行检索
    bio_data_list = bio_pubmed.search_bio_faiss(
        vector_query=base_query,
        original_query=current_search_query
    )
    # 3.3 保存文献证据
    if bio_data_list:
        for paper in bio_data_list:
            formatted_paper_str = refiner.format_single_paper(paper)
            if formatted_paper_str:
                meta = {
                    "title": paper.get("title", "Unknown"),
                    "year": paper.get("year", "Unknown"),
                    "id": paper.get("pmid", "Unknown")
                }
                final_evidence_list.append({
                    "source_type": "PubMed Literature",
                    "content": formatted_paper_str,
                    "metadata": meta
                })
    return final_evidence_list
