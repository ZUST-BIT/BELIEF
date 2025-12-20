# 对证据进行精炼和格式化，便于后续使用
class EvidenceRefiner:
    def __init__(self):
        pass

    def _clean_text(self, text):
        """清理文本，去掉多余换行和空格"""
        if not text:
            return ""
        return " ".join(text.split()).strip()

    def _format_text_data(self, data_list, source_name="Literature"):
        """
        优化后的文献证据格式：
        - 显式 Title
        - 精炼 Abstract（去掉 Title/Abstract 标签）
        - 保持多文献边界清晰
        """

        if not data_list:
            return f"No {source_name} data found."

        formatted = [f"--- {source_name} Evidence ---\n"]

        for idx, item in enumerate(data_list, start=1):
            title = self._clean_text(item.get("title", "Unknown Title"))
            raw_text = item.get("content") or item.get("abstract") or ""
            abstract = self._clean_text(raw_text)

            # 去掉冗余前缀（Title: / Abstract:）
            for prefix in ["Title:", "Abstract:", "Background:", "Methods:", "Results:", "Conclusion:"]:
                abstract = abstract.replace(prefix, "").strip()

            # 控制摘要长度，避免 LLM 过载（可按需要调）
            if len(abstract) > 1200:
                abstract = abstract[:1200].rsplit(" ", 1)[0] + "..."

            formatted.append(f"[{idx}] Title: {title}")
            formatted.append("Summary:")
            formatted.append(abstract)
            formatted.append("")  # 文献间空行

        return "\n".join(formatted).strip()

    def _format_omic_data(self, data_list):
        """格式化组学数据"""
        if not data_list:
            return "No Omic data found."
        formatted_str = "--- Omic Data Evidence ---\n"
        for idx, item in enumerate(data_list, start=1):
            text = self._clean_text(item.get("text", ""))
            formatted_str += f"{idx}. {text}\n\n"
        return formatted_str.strip()

    def _format_kg_data(self, kg_data):
        """
        将 Knowledge Graph 路径格式化为「推理路径 + 三元组链」
        目标：LLM 易理解、推理受控、信息不冗余
        """

        if not kg_data:
            return "No knowledge graph evidence found."

        output = []

        for key, path_info in kg_data.items():
            output.append(f"### {key}\n")

            # 防御式检查
            if not isinstance(path_info, dict):
                output.append(f"- {str(path_info)}\n")
                continue

            nodes = path_info.get("path_nodes", [])
            rels = path_info.get("path_rels", [])
            hops = path_info.get("length", len(rels))

            if not nodes or not rels:
                output.append("- Incomplete reasoning path.\n")
                continue

            output.append(f"Reasoning Path ({hops} hops):\n")

            for i, rel in enumerate(rels):
                src_node = nodes[i]
                dst_node = nodes[i + 1]

                def fmt_node(n):
                    name = n.get("properties", {}).get("name", "Unknown")
                    label = n.get("labels", ["Unknown"])[0]
                    source = n.get("properties", {}).get("source", "Unknown")
                    return f"{name} [{label} | {source}]"

                src = fmt_node(src_node)
                dst = fmt_node(dst_node)
                r_type = rel.get("type", "RelatedTo")

                output.append(
                    f"{i + 1}. ({src})\n"
                    f"   --[{r_type}]-->\n"
                    f"   ({dst})\n"
                )

        return "\n".join(output).strip()


    def run(self, kg_data, bio_data, omic_data=None, hcc_data=None):
        """
        主入口：整合所有证据
        kg_data: dict (Knowledge Graph)
        bio_data: list (PubMed 文献)
        omic_data: list (可选)
        hcc_data: list (可选)
        """
        evidence_sections = []

        # KG 数据
        kg_str = self._format_kg_data(kg_data)
        evidence_sections.append(kg_str)

        # Bio 数据
        bio_str = self._format_text_data(bio_data, source_name="PubMed Literature")
        evidence_sections.append(bio_str)

        # Omic 数据
        if omic_data:
            omic_str = self._format_omic_data(omic_data)
            evidence_sections.append(omic_str)

        # HCC 数据
        if hcc_data:
            hcc_str = self._format_text_data(hcc_data, source_name="HCC Clinical Literature")
            evidence_sections.append(hcc_str)

        # 拼接所有证据
        full_evidence = "\n\n".join(evidence_sections)
        return full_evidence