# utils/data_refiner.py

class EvidenceRefiner:
    def __init__(self):
        pass

    def clean_text(self, text):
        """[公有方法] 清理文本，去掉多余换行和空格"""
        if not text:
            return ""
        # 替换掉多余的换行符和连续空格
        return " ".join(str(text).split()).strip()

    def format_single_paper(self, paper_data):
        """
        [新增] 格式化单篇文献，返回一个清洗后的字符串
        """
        if not paper_data:
            return None

        # 1. 清洗标题和正文
        title = self.clean_text(paper_data.get("title", "Unknown Title"))
        raw_content = paper_data.get("content") or paper_data.get("abstract") or ""
        abstract = self.clean_text(raw_content)

        # 2. 去掉常见的冗余前缀，节省 Token
        prefixes_to_remove = ["Title:", "Abstract:", "Background:", "Methods:", "Results:", "Conclusion:"]
        for prefix in prefixes_to_remove:
            # 注意：这里简单的 replace 可能会误伤，建议只去掉开头的
            if abstract.startswith(prefix):
                abstract = abstract[len(prefix):].strip()

        # 3. 控制长度（防止单篇文献过长占满上下文）
        if len(abstract) > 1200:
            abstract = abstract[:1200].rsplit(" ", 1)[0] + "..."

        # 4. 组装成便于 LLM 阅读的格式
        # 格式：Title: <title>\nSummary: <abstract>
        formatted_str = f"Title: {title}\nSummary: {abstract}"
        return formatted_str

    def format_kg_data(self, kg_data):
        """
        [公有方法] 格式化知识图谱路径
        """
        if not kg_data:
            return "No knowledge graph evidence found."

        output = []
        for key, path_info in kg_data.items():
            output.append(f"### Entity: {key}")

            if not isinstance(path_info, dict):
                output.append(f"- {str(path_info)}")
                continue

            nodes = path_info.get("path_nodes", [])
            rels = path_info.get("path_rels", [])
            
            if not nodes or not rels:
                # output.append("- Incomplete reasoning path.") 
                continue # 如果路径不完整，直接跳过，减少噪音

            # 格式化路径: Node1 --[Rel]--> Node2
            path_str_list = []
            for i, rel in enumerate(rels):
                if i + 1 >= len(nodes): break
                
                src = nodes[i].get("properties", {}).get("name", "Unknown")
                dst = nodes[i+1].get("properties", {}).get("name", "Unknown")
                rel_type = rel.get("type", "RelatedTo")
                
                path_str_list.append(f"({src}) -[{rel_type}]-> ({dst})")
            
            output.append("\n".join(path_str_list))
            output.append("") # 空行分隔

        return "\n".join(output).strip()