class EvidenceRefiner:
    def __init__(self):
        pass

    def _format_text_data(self, data_list, source_name="Literature"):
        """
        处理 HCC 和 Bio (PubMed) 这种文献类数据
        格式目标:
        1. Title: xxx
           Section: xxx (如果有)
           Content: xxx
        """
        if not data_list:
            return "No data found."

        formatted_str = f"--- {source_name} Evidence ---\n"
        
        for idx, item in enumerate(data_list):
            # 1. 提取 Title
            title = item.get("title", "Unknown Title").strip()
            
            # 2. 提取 Section (HCC数据通常有，PubMed可能没有)
            section = item.get("section", "")
            
            # 3. 提取 Content (优先找 content，没有找 abstract)
            content = item.get("content") or item.get("abstract") or ""
            # 清洗一下 content，去掉过多的换行，让它紧凑一点
            content = content.replace("\n", " ").strip()

            # 4. 组装字符串
            formatted_str += f"{idx + 1}. Title: {title}\n"
            if section:
                formatted_str += f"   Section: {section}\n"
            formatted_str += f"   Content: {content}\n\n"
            
        return formatted_str.strip()

    def _format_omic_data(self, data_list):
        """
        处理组学数据
        Omic 数据里的 'text' 字段已经是整理好的描述，直接提取即可
        """
        if not data_list:
            return "No Omic data found."

        formatted_str = "--- Omic Data Evidence ---\n"
        
        for idx, item in enumerate(data_list):
            # 直接获取 text 字段
            text = item.get("text", "").strip()
            formatted_str += f"{idx + 1}. {text}\n\n"
            
        return formatted_str.strip()

    def _format_kg_data(self, kg_data):
        """
        处理 KG 数据
        直接罗列关系，不做复杂清洗
        """
        if not kg_data:
            return "No Knowledge Graph data found."

        formatted_str = "--- Knowledge Graph Evidence ---\n"
        
        # kg_data 通常是 {'Lenvatinib one-hop info': [...list...]}
        if isinstance(kg_data, dict):
            for key, relations in kg_data.items():
                formatted_str += f"[{key}]:\n"
                for rel in relations:
                    formatted_str += f"- {rel}\n"
                formatted_str += "\n"
        else:
            # 如果结构不一样，直接转字符串
            formatted_str += str(kg_data)

        return formatted_str.strip()

    def run(self, kg_data, omic_data, hcc_data, bio_data):
        """
        主入口：接收四个原始列表，返回一个拼接好的大字符串
        """
        # 1. 格式化 HCC 数据
        hcc_str = self._format_text_data(hcc_data, source_name="HCC Clinical Literature")
        
        # 2. 格式化 Bio 数据
        bio_str = self._format_text_data(bio_data, source_name="PubMed General Literature")
        
        # 3. 格式化 Omic 数据
        omic_str = self._format_omic_data(omic_data)
        
        # 4. 格式化 KG 数据 (保持原样)
        kg_str = self._format_kg_data(kg_data)

        # 5. 拼接所有内容
        # 这个最终的 full_evidence 就是要喂给 Coder 的 Prompt 的内容
        full_evidence = (
            f"{kg_str}\n\n"
            f"{omic_str}\n\n"
            f"{hcc_str}\n\n"
            f"{bio_str}"
        )
        
        return full_evidence
