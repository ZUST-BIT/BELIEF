"""
智能体工具模块
"""
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import set_argument
from prompt import evaluator_prompt, analyst_prompt, generate_prompt_pubmedqa

# 全局配置（延迟加载）
_config = None
_llm = None

def _get_config():
    """获取配置（延迟加载）"""
    global _config
    if _config is None:
        _config = set_argument()
    return _config

def _get_llm(model_name: str = "gpt-4o-mini"):
    """获取LLM实例（延迟加载）"""
    global _llm
    if _llm is None:
        config = _get_config()
        _llm = ChatOpenAI(
            model=model_name,
            api_key=config.api_key_gpt,
            base_url=config.api_url_gpt,
            temperature=0
        )
    return _llm
    
# 答案生成器 - 负责生成最终答案
class GeneratorAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        初始化生成器代理
        
        Args:
            model_name: 模型名称
        """
        config = _get_config()
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=config.api_key_gpt,
            base_url=config.api_url_gpt
        )

    def generate_answer(self, query, evidence_text):
        formatted_prompt = generate_prompt_pubmedqa.format(
            question=query, 
            evidence=evidence_text
        )

        messages = []
        user_content = []
        
        # 将填充好的完整提示词作为文本输入
        user_content.append({"type": "text", "text": formatted_prompt})    
        messages.append(("user", user_content))
        response = self.llm.invoke(messages)
        return response.content

# 证据评估者 - 负责评估证据是否足以回答问题
class EvidenceEvaluator:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        初始化反思者代理
        
        Args:
            model_name: 模型名称
        """
        config = _get_config()
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0, # 评判需要冷静，不要随机性
            api_key=config.api_key_gpt,
            base_url=config.api_url_gpt
        )

    def evaluate_global(self, query: str, evidence_list: list):
        """
        [新功能] 宏观评估：基于证据列表判断是否 Pass/Fail
        对应 Prompt: evaluator_prompt (上一轮提供的新版)
        """
        # 将证据列表序列化为 JSON 字符串
        evidence_json_str = json.dumps(evidence_list, ensure_ascii=False, indent=2)

        template = PromptTemplate(
            input_variables=["question", "evidence_list_json"],
            template=evaluator_prompt
        )
        chain = template | self.llm | StrOutputParser()
        
        try:
            res_str = chain.invoke({
                "question": query, 
                "evidence_list_json": evidence_json_str
            })
            # 清洗 markdown
            cleaned = res_str.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except Exception as e:
            print(f"⚠️ Reflector Error: {e}")
            return {
                "status": "fail", 
                "report": {"missing_info": "System Error in Evaluation"}, 
                "next_step_strategy": {"search_query": query}
            }
        
# 证据多维度分析者 - 负责分析证据
class MuldimAnalyst:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        config = _get_config()
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0, # 分析必须客观
            api_key=config.api_key_gpt,
            base_url=config.api_url_gpt
        )

    def analyze_single(self, query: str, evidence_item: dict, focus_instruction: str = ""):
        """
        [新功能] 逐条分析：对单个证据块打分和提取
        对应 Prompt: analyst_prompt (上一轮提供的新版)
        """
        context_str = evidence_item.get("content", "")
        # 如果内容太短，直接忽略，节省 Token
        if len(context_str) < 15:
            return None

        template = PromptTemplate(
            input_variables=["question", "context", "focus_instruction"],
            template=analyst_prompt
        )
        chain = template | self.llm | StrOutputParser()

        try:
            res_str = chain.invoke({
                "question": query,
                "context": context_str,
                "focus_instruction": focus_instruction
            })
            cleaned = res_str.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except Exception as e:
            print(f"⚠️ Analyst Error (Item ID: {evidence_item.get('metadata', {}).get('id', 'unknown')}): {e}")
            return None