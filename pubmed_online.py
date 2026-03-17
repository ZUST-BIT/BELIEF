# pubmed_online.py
"""
NCBI PubMed E-utilities 在线检索模块
使用 NCBI Entrez API 直接检索 PubMed 文献，无需本地索引
API 文档: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import re
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

# NCBI E-utilities 基础 URL
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# 建议填写自己的邮箱，以便 NCBI 在出现问题时联系（遵守其使用条款）
NCBI_TOOL  = "MEDAR-QA"
NCBI_EMAIL = "qq384438241@gmail.com"


class PubMedOnlineSearcher:
    """
    封装 NCBI E-utilities esearch + efetch 流程，
    输入查询字符串，返回格式化后的文献列表。
    """

    def __init__(
        self,
        top_k: int = 5,
        max_abstract_len: int = 1200,
        request_timeout: int = 15,
        retry: int = 2,
    ):
        """
        Args:
            top_k: 返回文献数量上限
            max_abstract_len: 摘要最大字符数（超出时截断）
            request_timeout: 单次 HTTP 请求超时时间（秒）
            retry: 请求失败时的重试次数
        """
        self.top_k = top_k
        self.max_abstract_len = max_abstract_len
        self.request_timeout = request_timeout
        self.retry = retry

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        主入口：给定自然语言查询，返回 PubMed 文献列表。

        Returns:
            [
                {
                    "pmid": "12345678",
                    "title": "...",
                    "abstract": "...",
                    "year": "2023",
                    "authors": "...",
                }
            ]
        """
        pmids = self._esearch(query, retmax=self.top_k)
        if not pmids:
            return []

        papers = self._efetch(pmids)
        return papers

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict) -> Optional[requests.Response]:
        """带重试的 GET 请求"""
        for attempt in range(self.retry + 1):
            try:
                resp = requests.get(url, params=params, timeout=self.request_timeout)
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt < self.retry:
                    time.sleep(1.0)
                else:
                    print(f"[PubMedOnline] 请求失败（{url}）: {e}")
                    return None

    # PubMed Boolean 搜索常见停用词（虚词/系动词），保留后会严重干扰匹配
    # 注意：不包含 'or'/'and'，这两个词在 PubMed 中是布尔运算符，需要保留
    _STOP_WORDS = {
        'a', 'an', 'the', 'is', 'it', 'its', 'are', 'was', 'were', 'be',
        'been', 'being', 'do', 'does', 'did', 'have', 'has', 'had',
        'in', 'on', 'at', 'to', 'of', 'for', 'by', 'but',
        'not', 'with', 'from', 'as', 'this', 'that', 'which', 'what',
        'who', 'how', 'why', 'when', 'where', 'if', 'whether',
    }

    def _clean_query(self, query: str) -> str:
        """
        将自然语言问句转换为 PubMed 关键词风格查询。

        PubMed E-utilities 是 Boolean 关键词搜索，不做 NLP，因此：
          1. 去除特殊运算符（:  []  "  ?）
          2. 过滤虚词/停用词
          3. 限制词数（≤12词），超出部分截去
        """
        # 1. 去掉已知 PubMed 特殊符号
        cleaned = re.sub(r'[:\[\]"?()]', ' ', query)
        # 2. 多个空格合并
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # 3. 过滤停用词，保留医学关键词
        words = [
            w for w in cleaned.split()
            if w.lower() not in self._STOP_WORDS and len(w) > 1
        ]
        # 4. 限制在 12 个词以内，防止 PubMed 把长句当短语匹配
        words = words[:12]
        return ' '.join(words)

    def _esearch(self, query: str, retmax: int) -> List[str]:
        """
        调用 esearch 获取 PMID 列表
        使用 [tiab] 限定在标题和摘要中检索，提高相关性
        """
        clean = self._clean_query(query)
        params = {
            "db": "pubmed",
            "term": clean,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
            "tool": NCBI_TOOL,
            "email": NCBI_EMAIL,
        }
        resp = self._get(ESEARCH_URL, params)
        if resp is None:
            return []
        try:
            data = resp.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            return pmids
        except Exception as e:
            print(f"[PubMedOnline] esearch 解析失败: {e}")
            return []

    def _efetch(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        调用 efetch 获取摘要，返回格式化文献列表
        """
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "tool": NCBI_TOOL,
            "email": NCBI_EMAIL,
        }
        resp = self._get(EFETCH_URL, params)
        if resp is None:
            return []

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as e:
            print(f"[PubMedOnline] XML 解析失败: {e}")
            return []

        papers = []
        for article in root.iter("PubmedArticle"):
            paper = self._parse_article(article)
            if paper:
                papers.append(paper)
        return papers

    def _parse_article(self, article: ET.Element) -> Optional[Dict[str, Any]]:
        """从单篇 PubmedArticle XML 节点提取所需字段"""
        # PMID
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None else "Unknown"

        # 标题
        title_el = article.find(".//ArticleTitle")
        title = self._get_text(title_el)

        # 摘要（可能有多个 AbstractText 段落）
        abstract_parts = []
        for abs_el in article.findall(".//AbstractText"):
            label = abs_el.get("Label", "")
            text = self._get_text(abs_el)
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts).strip()

        # 截断摘要
        if len(abstract) > self.max_abstract_len:
            abstract = abstract[: self.max_abstract_len].rsplit(" ", 1)[0] + "..."

        # 年份
        year_el = article.find(".//PubDate/Year")
        year = year_el.text.strip() if year_el is not None else "Unknown"

        # 作者（仅取姓氏列表，节省 token）
        authors = []
        for author_el in article.findall(".//Author"):
            last = author_el.find("LastName")
            if last is not None and last.text:
                authors.append(last.text)
        authors_str = ", ".join(authors[:5])
        if len(authors) > 5:
            authors_str += " et al."

        if not title and not abstract:
            return None

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "year": year,
            "authors": authors_str,
        }

    @staticmethod
    def _get_text(element: Optional[ET.Element]) -> str:
        """递归提取 XML 元素的所有文本内容"""
        if element is None:
            return ""
        return "".join(element.itertext()).strip()
