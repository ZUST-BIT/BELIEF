# retriever.py
import json
from typing import Dict, Any, List
from neo4j import Neo4jManager
from pubmed import bio_pubmed_data
from utils.data_refiner import EvidenceRefiner
from utils.entity_linker import get_entity_retriever_instance

def retrieve_process(
    current_search_query: str, 
    agent_a_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    基于智能体A的分析结果执行 RAG 流程 (Hybrid Search)
    
    Args:
        current_search_query: 用户的原始问题
        agent_a_result: 智能体A的分析结果，包含实体列表、问题类型等信息
    
    Returns:
        检索到的证据列表
    """
    # 1. 初始化模块（使用单例模式，避免重复加载模型）
    neo4j_manager = Neo4jManager()
    bio_pubmed = bio_pubmed_data()
    refiner = EvidenceRefiner()
    entity_linker = get_entity_retriever_instance()
    
    final_evidence_list = []
    entities_to_align = []
    
    # 智能体A可能返回的实体字段名称
    entity_keys = ['biomedical_entities', 'entities', 'key_entities', 'extracted_entities']
    
    for key in entity_keys:
        if key in agent_a_result:
            entities = agent_a_result[key]
            if isinstance(entities, list):
                entities_to_align.extend(entities)
            elif isinstance(entities, dict):
                # 如果实体是字典格式，提取值
                for entity_list in entities.values():
                    if isinstance(entity_list, list):
                        entities_to_align.extend(entity_list)
            break
    
    # 如果智能体A提取了PICO元素，也加入检索
    if 'pico' in agent_a_result:
        pico = agent_a_result['pico']
        for element in ['P', 'I', 'C', 'O']:
            if element in pico and pico[element]:
                value = pico[element]
                if isinstance(value, str) and value not in ["null", "N/A", ""]:
                    entities_to_align.append(value)
                elif isinstance(value, list):
                    entities_to_align.extend(value)
    
    # 去重和清洗
    entities_to_align = list(set([
        e.strip() if isinstance(e, str) else str(e) 
        for e in entities_to_align 
        if e and str(e).strip() and str(e).strip() not in ["null", "N/A", ""]
    ]))
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

    # 3. PubMed 文献检索
    # print("\n📚 [PubMed Search] 检索相关文献...")
    
    # 构建检索查询（使用原始问题 + 扩展词）
    base_query = current_search_query
    
    # 简单去重并拼接扩展词
    unique_expansion = list(set(expansion_terms))
    filtered_expansion = [t for t in unique_expansion if t.lower() not in base_query.lower()]
    
    final_vector_query = f"{base_query} {' '.join(filtered_expansion[:5])}".strip()
    # print(f"   检索查询: {final_vector_query}")
    
    # 执行检索
    bio_data_list = bio_pubmed.search_bio_faiss(
        vector_query=final_vector_query,
        original_query=current_search_query
    )
    
    # 保存文献证据
    if bio_data_list:
        # print(f"   检索到 {len(bio_data_list)} 篇文献")
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
    else:
        print("   未检索到相关文献")
    
    # print(f"\n✅ 检索完成，共获得 {len(final_evidence_list)} 个证据片段")
    return final_evidence_list
