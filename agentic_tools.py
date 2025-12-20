# 整体智能体代码
import json
import os
import base64
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from format_config import FormatType, get_format_instructions
from config import set_argument
from prompt import reasoner_prompt, evaluator_prompt
model_name = "gpt-4o-mini"
args = set_argument()
llm = ChatOpenAI(model=model_name, api_key=args.api_key_gpt, base_url=args.api_url_gpt, temperature=0)
# 1. 格式选择器 The Router
def format_selector(query, evidence_text):
    format_instructions_text = get_format_instructions()
    system_prompt = f"""
    你是一个生物医学数据的展示规划师。
    根据用户的问题和提供的证据数据，选择一种最佳的展示格式。
    可用格式及适用场景:
    {format_instructions_text}
        
    判断逻辑补充:
    1. 哪怕用户问了趋势，但如果证据里全是文字没有数字，必须降级为 TEXT。
    2. 如果证据包含 'Omic Data' 里的 IC50/AUC 数值，优先考虑 CHART。
    3. 如果证据包含 'KG Data' 里的实体关系，优先考虑 GRAPH。
    
    请仅返回 JSON 格式: {{"selected_format": "...", "reasoning": "..."}}
    其中 selected_format 必须是上述可用格式的名称之一 (例如 'chart', 'graph', 'text')。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "User Query: {query}\n\nRefined Evidence Preview:\n{evidence}...")
    ])
        
    chain = prompt | llm
    try:
        res = chain.invoke({"query": query, "evidence": evidence_text[:3000],"format_instructions_text": format_instructions_text})
        content = res.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"Format selection failed: {e}")
        # 出错时默认回退到 TEXT
        return {"selected_format": FormatType.TEXT.value, "reasoning": "Error in selection"}

# 2. Coder Agent - 负责生成可执行代码
def generate_code(format_type, evidence_text):
    system_prompt = """
        你是一个 Python 数据可视化专家。
        你的任务是：解析提供的文本证据，提取数据，并编写 Python 代码进行绘图。
        
        要求:
        1. **数据提取**: 你必须从提供的文本中通过正则表达式或字符串处理提取数据。
           - 严禁读取外部 CSV/Excel 文件。数据必须硬编码在代码的 List 或 DataFrame 中。
        2. **绘图库**: 
           - 'chart' 使用 matplotlib/seaborn。
           - 'graph' 使用 networkx 和 matplotlib。
        3. **样式优化**: 
           - 使用 seaborn 默认样式。
           - 标题和标签尽量使用英文，防止中文乱码。
        4. **输出**: 代码必须保存图片到当前目录，文件名为 'visualization_result.png'。
        5. **格式**: 只返回 Python 代码，不要包含 Markdown 标记。
        """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Target Format: {format_type}\n\nEvidence Data:\n{evidence}")
    ])
    
    chain = prompt | llm
    
    res = chain.invoke({
        "format_type": format_type,
        "evidence": evidence_text
    })

    return res.content.replace("```python", "").replace("```", "").strip()

# 3. 执行器 - 负责执行代码并生成结果图片
class CodeExecutor:
    def execute(self, code):
        try:
            print(">>> Executing Visualization Code...")
            exec_globals = {}
            exec(code, exec_globals)
            if os.path.exists('visualization_result.png'):
                print(">>> Visualization saved as 'visualization_result.png'")
                return "visualization_result.png"
            else:
                print(">>> Code executed but file not found.")
                return None
        except Exception as e:
            print(f">>> Code Execution Failed: {e}")
            return None
        
# 4. 生成器 - 负责生成最终答案
class GeneratorAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        args = set_argument()
        api_key = args.api_key_gpt
        base_url = args.api_url_gpt
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1, 
            api_key=api_key,
            base_url=base_url
        )

    def _encode_image(self, image_path):
        if not image_path or not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_answer(self, query, evidence_text, image_path=None):
        print(">>> Generator: Synthesizing final answer...")
        messages = []
        system_content = """
        你是一个专业的生物医学研究助手。请根据提供的文本证据和数据图表，回答用户的问题。
        要求：
        1. **图文结合**：如果提供了图表，请在回答中明确引用图表内容（例如“如图所示，Lenvatinib的IC50值...”）。
        2. **严谨性**：只根据提供的证据回答，不要编造事实。
        3. **结构化**：回答要有逻辑，分点陈述。
        """
        messages.append(("system", system_content))
        
        user_content = []
        user_content.append({"type": "text", "text": f"用户问题: {query}\n\n文本证据摘要:\n{evidence_text}"})
        
        if image_path:
            base64_image = self._encode_image(image_path)
            if base64_image:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
        
        messages.append(("user", user_content))
        response = self.llm.invoke(messages)
        return response.content

# 5. 多模态上下文 - 管理文本证据和视觉产物
class MultimodalContext:
    def __init__(self):
        self.textual_evidence = [] # 存储每一轮的文本证据
        self.visual_artifacts = [] # 存储生成的图片路径
        self.query_history = []    # 存储搜索过的关键词，防止重复搜索
        self.reasoning_traces = [] # 存储推理过程记录，用于追溯推理链路
    def add_evidence(self, text):
        if text and len(text) > 10: # 忽略太短的无效信息
            for existing in self.textual_evidence:
                if text in existing:
                    return
            self.textual_evidence.append(text)
            
    def add_image(self, image_path):
        if image_path and image_path not in self.visual_artifacts:
            self.visual_artifacts.append(image_path)
            
    def add_query(self, query):
        if query not in self.query_history:
            self.query_history.append(query)

    def add_reasoning(self, reasoning_text):
        if reasoning_text:
            self.reasoning_traces.append(reasoning_text)

    def get_consolidated_text(self):
        """
        合并：原始证据 + 推理出的洞察
        """
        content_parts = []
        
        # 1. 原始证据
        if self.textual_evidence:
            for i, t in enumerate(self.textual_evidence):
                content_parts.append(f"--- Evidence Batch {i+1} ---\n{t}")
        
        # 2. 推理洞察 (加在后面，作为高层级总结)
        if self.reasoning_traces:
            content_parts.append("\n=== 🧠 LOGICAL REASONING / INSIGHTS ===")
            for i, r in enumerate(self.reasoning_traces):
                content_parts.append(f"[Insight {i+1}]: {r}")
                
        return "\n\n".join(content_parts)
    
    def show_status(self):
        print("\n" + "="*30)
        print("🧠 Multimodal Context Status")
        print("="*30)
        print(f"🔍 Query History ({len(self.query_history)})")
        print(f"📚 Evidence Batches ({len(self.textual_evidence)})")
        print(f"💡 Reasoning Traces ({len(self.reasoning_traces)})") # 打印推理数量
        print(f"🖼️  Visual Artifacts ({len(self.visual_artifacts)})")
        print("="*30 + "\n")

# 6. 推理者- 负责分析证据间的逻辑联系，并生成推理链路
class ReasonerAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        # 自动加载配置
        args = set_argument()
        api_key = args.api_key_gpt
        base_url = args.api_url_gpt
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            api_key=api_key,
            base_url=base_url
        )

    def analyze(self, query, user_context,full_evidence):
        """
        深度推理
        Args:
            query: 用户问题
            user_context: 用户提供的背景文本（Ground Truth）
            full_evidence: 多轮检索得到的所有证据文本
        """
        user_input_str = f"""
        User Query: {query}
        User Context: {user_context if user_context else 'No additional context provided.'}
        Retrieved Evidence:
        {full_evidence if full_evidence else 'No evidence retrieved yet.'}
        """

        messages = [
            ("system", reasoner_prompt),
            ("user", user_input_str)
        ]
        try:
            res = self.llm.invoke(messages)
            return res.content
        except Exception as e:
            print(f"Reasoner Error: {e}")
            return "Error in reasoning process."

# 7. 反思者 - 负责评估证据是否足以回答问题
class ReflectorAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        # 自动加载配置
        args = set_argument()
        api_key = args.api_key_gpt
        base_url = args.api_url_gpt
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0, # 评判需要冷静，不要随机性
            api_key=api_key,
            base_url=base_url
        )

    def evaluate(self, original_query, consolidated_evidence):
        """
        评估证据是否足以回答问题。
        返回: {"status": "pass" 或 "fail", "feedback": "...", "new_query": "..."}
        """
        system_prompt = """
        你是一个严格的生物医学研究导师。你的任务是评估【当前积累的所有证据】是否足以回答用户的【原始问题】。
        
        请执行以下检查：
        1. **信息覆盖度**：用户问题中的关键实体（如特定药物、基因、副作用、机制）是否在证据中都有涉及？
        2. **对比完整性**：如果用户要求“对比”或“趋势”，证据中是否包含相关的数值或结论？
        3. **逻辑闭环**：现有的碎片信息能否拼凑出一个完整的逻辑链条？
        
        输出规则：
        - 如果证据已经足够回答问题，status 为 "pass"。
        - 如果缺少关键信息（例如：只查到了药物A，没查到药物B），status 为 "fail"。
        - 如果 status 为 "fail"，必须提供 new_query，这是为了获取缺失信息而生成的**新的、具体的搜索关键词**。
        
        请仅返回 JSON 格式: 
        {{
            "status": "pass/fail", 
            "feedback": "简短评价当前缺失了什么信息", 
            "new_query": "用于下一轮搜索的关键词 (如果 pass 则留空)"
        }}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "User Original Query: {query}\n\nAll Accumulated Evidence:\n{evidence}")
        ])
        
        chain = prompt | self.llm
        
        try:
            res = chain.invoke({
                "query": original_query, 
                "evidence": consolidated_evidence[:6000] # 限制长度，防止上下文爆炸
            })
            content = res.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"Reflector Error: {e}")
            # 出错时为了防止死循环，保守起见默认通过
            return {"status": "pass", "feedback": "Error parsing decision", "new_query": ""}
        
# 8. 证据多维度评估者 - 负责评估证据评估
class MultiEvaluator:
    def __init__(self,model_name="gpt-4o-mini"):
        args = set_argument()
        api_key = args.api_key_gpt
        base_url = args.api_url_gpt
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key,
            base_url=base_url
        )
    def evaluate(self, query, context, metadata=None):
        """
        多维度证据分析
        Args：
            query: 用户问题
            context: 多轮检索得到的所有证据文本
            metadata: 元数据(可选)，包括原始问题、问题类型、问题背景、问题
        """
        if metadata is None:
            metadata = "No additional metadata provided."
        template = PromptTemplate(
            input_variables = ["question", "context", "metadata"],
            template = evaluator_prompt
        )
        parser = StrOutputParser()
        chain = template | self.llm | parser
        try:
            res = chain.invoke({
                "question": query,
                "context": context,
                "metadata": metadata
            })
            return res
        except Exception as e:
            print(f"Evaluator Error: {e}")
            return "Error in evaluation process."



    