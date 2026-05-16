# retriever.py
from typing import Dict, Any, List
from pubmed_online import PubMedOnlineSearcher

# 单例：避免每次调用都重新初始化
_pubmed_searcher: PubMedOnlineSearcher = None


def _get_pubmed_searcher() -> PubMedOnlineSearcher:
    global _pubmed_searcher
    if _pubmed_searcher is None:
        _pubmed_searcher = PubMedOnlineSearcher(top_k=5)
    return _pubmed_searcher


def _extract_clean_keywords(agent_a_result: Dict[str, Any]) -> List[str]:
    """
    从智能体A的 search_strategy.primary_keywords 提取干净的关键词列表。
    返回空列表表示不可用。
    """
    search_strategy = agent_a_result.get("search_strategy", {})
    primary_keywords = search_strategy.get("primary_keywords", [])
    if not isinstance(primary_keywords, list):
        return []
    return [
        str(kw).strip()
        for kw in primary_keywords
        if kw and str(kw).strip() not in ("null", "N/A", "")
    ]


def _dedup_keywords(keywords: List[str]) -> List[str]:
    """按小写去重并保留原顺序。"""
    seen = set()
    deduped = []
    for kw in keywords:
        key = kw.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(kw.strip())
    return deduped


def _to_tiab_term(keyword: str) -> str:
    """将关键词转换为 PubMed Title/Abstract 字段检索项。"""
    kw = keyword.replace('"', ' ').strip()
    if not kw:
        return ""
    if " " in kw:
        return f'"{kw}"[Title/Abstract]'
    return f'{kw}[Title/Abstract]'


def _build_staged_queries(keywords: List[str]) -> List[str]:
    """
    构建分层检索 query：
      1) 精准层：核心词 AND 扩展词(OR)
      2) 召回层：全部词 OR
    """
    if not keywords:
        return []

    kws = _dedup_keywords(keywords)
    tiab_terms = [_to_tiab_term(kw) for kw in kws]
    tiab_terms = [t for t in tiab_terms if t]
    if not tiab_terms:
        return []

    # 常见做法：将前2个最重要关键词作为核心锚点
    core_terms = tiab_terms[:2]
    support_terms = tiab_terms[2:8]

    queries = []
    if core_terms and support_terms:
        strict_query = f"({' OR '.join(core_terms)}) AND ({' OR '.join(support_terms)})"
        queries.append(strict_query)

    # 当关键词很少时，核心层也可作为单独检索
    if core_terms:
        core_only_query = " OR ".join(core_terms)
        queries.append(core_only_query)

    broad_query = " OR ".join(tiab_terms)
    queries.append(broad_query)

    # 去重并保序
    uniq_queries = []
    seen = set()
    for q in queries:
        if q not in seen:
            seen.add(q)
            uniq_queries.append(q)
    return uniq_queries


def retrieve_process(
    current_search_query: str,
    agent_a_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    基于智能体A的分析结果执行 PubMed 在线关键词检索。

        检索策略：采用“先精确后召回”的分层检索。

    流程：
            1. 主路：基于关键词构建 Title/Abstract 字段检索，先走精准层，再走召回层
            2. 兜底：若关键词检索无结果，使用原始问题句（内部自动清洗）

    Args:
        current_search_query: 用户的原始问题
        agent_a_result: 智能体A的分析结果（包含 search_strategy 字段）

    Returns:
        检索到的文献证据列表
    """
    final_evidence_list = []
    searcher = _get_pubmed_searcher()
    papers = []
    source_label = "PubMed Literature (Keyword Search)"

    clean_kws = _extract_clean_keywords(agent_a_result)

    if clean_kws:
        staged_queries = _build_staged_queries(clean_kws)
        labels = [
            "PubMed Literature (Keyword Strict Search)",
            "PubMed Literature (Keyword Core Search)",
            "PubMed Literature (Keyword Broad Search)",
        ]
        for idx, query in enumerate(staged_queries):
            # print(f"[Retriever] PubMed Query (阶段 {idx + 1}/{len(staged_queries)}): {query}")
            papers = searcher.search(query)
            if papers:
                source_label = labels[idx] if idx < len(labels) else "PubMed Literature (Keyword Search)"
                # print(f"[Retriever] 命中分层检索阶段 {idx + 1}/{len(staged_queries)}")
                break

    # 兜底：primary_keywords 不可用或仍无结果时，使用原始问题句
    if not papers:
        # print(f"[Retriever] 关键词检索无结果，降级到原始问题检索...")
        # print(f"[Retriever] PubMed Query (Fallback): {current_search_query}")
        papers = searcher.search(current_search_query)
        source_label = "PubMed Literature (Keyword Fallback)"

    # ----------------------------------------------------------------
    # 格式化并汇总结果
    # ----------------------------------------------------------------
    if papers:
        for paper in papers:
            abstract = paper.get("abstract", "")
            title    = paper.get("title", "Unknown Title")

            if len(abstract) > 1200:
                abstract = abstract[:1200].rsplit(" ", 1)[0] + "..."

            formatted_str = f"Title: {title}\nSummary: {abstract}"

            final_evidence_list.append({
                "source_type": source_label,
                "content": formatted_str,
                "metadata": {
                    "title":     title,
                    "year":      paper.get("year", "Unknown"),
                    "id":        paper.get("pmid", ""),
                    "authors":   paper.get("authors", ""),
                    "citations": paper.get("citations", 0),
                }
            })
    else:
        print("[Retriever] 未检索到相关文献")

    return final_evidence_list
