from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# 1. 定义支持的格式枚举
class FormatType(Enum):
    TEXT = "text"           # 纯文本解释
    LIST = "list"           # 结构化列表 (Markdown)
    TABLE = "table"         # 表格 (Markdown/Pandas)
    CHART = "chart"         # 统计图表 (Matplotlib/Seaborn)
    GRAPH = "graph"         # 知识图谱子图 (NetworkX)
    CODE = "code"           # 代码片段 (针对特定计算需求)

# 2. 定义每种格式的元数据（这就是你的 Format Knowledge）
class FormatDefinition(BaseModel):
    format_type: FormatType
    description: str
    suitable_scenarios: List[str]
    required_data_structure: str

# 3. 初始化表达形式库
FORMAT_KNOWLEDGE_BASE = {
    FormatType.TEXT: FormatDefinition(
        format_type=FormatType.TEXT,
        description="自然语言文本，用于叙述、概括或解释机制。",
        suitable_scenarios=["一般性回答", "摘要", "复杂概念解释"],
        required_data_structure="String or List[String]"
    ),
    FormatType.LIST: FormatDefinition(
        format_type=FormatType.LIST,
        description="项目符号列表，用于枚举实体或步骤。",
        suitable_scenarios=["列举药物", "列举基因", "推荐步骤", "副作用列表"],
        required_data_structure="List[String] or List[Dict]"
    ),
    FormatType.CHART: FormatDefinition(
        format_type=FormatType.CHART,
        description="统计图表（柱状图、折线图、散点图、热图）。",
        suitable_scenarios=["基因表达量趋势", "生存分析对比", "数值分布", "药物剂量反应"],
        required_data_structure="DataFrame (Structured Numerical Data)"
    ),
    FormatType.GRAPH: FormatDefinition(
        format_type=FormatType.GRAPH,
        description="节点-边 关系图。",
        suitable_scenarios=["信号通路", "蛋白质相互作用(PPI)", "药物-靶点关系", "疾病共病网络"],
        required_data_structure="List of Triples (Head, Relation, Tail)"
    )
}

# 辅助函数：生成LLM System Prompt
def get_format_instructions():
    instructions = "You have the following formats available to present the evidence:\n"
    for fmt_type, definition in FORMAT_KNOWLEDGE_BASE.items():
        instructions += f"- {fmt_type.value.upper()}: {definition.description}\n"
        instructions += f"  Use when: {', '.join(definition.suitable_scenarios)}\n"
    return instructions