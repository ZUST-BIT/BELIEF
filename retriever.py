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


def retrieve_process(
    current_search_query: str,
    agent_a_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    基于智能体A的分析结果执行 PubMed 在线关键词检索。

    检索策略：使用 OR 拼接 primary_keywords，最大化召回。
    PubMed 中 OR 表示「任意一个关键词命中即返回」，
    相比 AND 交集能检索到更多相关文献（保证 top_k 数量）。

    流程：
      1. 主路：primary_keywords 全部 OR 拼接，一次性检索
      2. 兜底：若 primary_keywords 不可用或无结果，使用原始问题句（内部自动清洗）

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
        # 用 OR 拼接所有关键词：任意一个命中即可返回结果，最大化召回
        or_query = " OR ".join(clean_kws)
        # print(f"[Retriever] OR 检索词: {or_query!r}")
        papers = searcher.search(or_query)

    # 兜底：primary_keywords 不可用或仍无结果时，使用原始问题句
    if not papers:
        print(f"[Retriever] OR 检索无结果，降级到原始问题检索...")
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
