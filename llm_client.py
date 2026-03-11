"""
LLM统一客户端模块
支持多种模型调用方式：
1. OpenAI兼容API（付费API、vLLM、Ollama等）
2. 本地Transformers加载
"""

import json
import requests
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


# ==================== 配置参数 ====================
# 选择使用的后端: "api" | "ollama" | "vllm" | "transformers"
LLM_BACKEND = "api"

# API配置（适用于 api 后端，付费API）
API_URL = "https://api.gptsapi.net/v1"
API_KEY = "sk-IGQ8241037bdbeccfb18105d4774dc98ac20067097dQ3dDL"
MODEL_NAME = "gpt-4o-mini"

# Ollama配置（本地运行，最简单）
# 安装: https://ollama.ai  然后运行: ollama pull qwen2.5:7b
OLLAMA_URL = "http://172.18.51.166:11434"
OLLAMA_MODEL = "qwen3:8b"  # 可选: "qwen2.5:14b", "llama3.1:8b", "deepseek-v2:16b",qwen3:14b

# vLLM配置（高性能推理服务器）
# 启动命令: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000
# 或直接运行 start_vllm.bat
VLLM_URL = "http://localhost:8000/v1"
VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 需要与启动时的模型一致

# 本地Transformers配置（直接加载，无需服务）
LOCAL_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"  # HuggingFace模型ID或本地路径
DEVICE = "cuda"  # "cuda" 或 "cpu"
# ================================================


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    @abstractmethod
    def chat(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """发送对话请求"""
        pass


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI兼容API客户端（支持OpenAI、vLLM、Ollama的OpenAI兼容模式）"""
    
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model = model
    
    def _remove_think_tags(self, text: str) -> str:
        """
        移除 Qwen3 等模型的思考标签 <think>...</think>
        处理三种情况：
        1. 正常闭合的 <think>...</think> 标签
        2. 只有 <think> 没有 </think>（max_tokens截断）
        3. 混合情况
        """
        import re
        if not text:
            return text

        # 情况1：标签正常闭合，移除全部 <think>...</think> 块
        if '</think>' in text:
            cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            cleaned = cleaned.strip()
            if cleaned:
                return cleaned
            # 若清理后为空，说明 </think> 后无内容，尝试降级：直接取最后一个 </think> 之后的文本
            after_think = text.split('</think>')[-1].strip()
            if after_think:
                return after_think
            # 真的没有任何输出内容（模型被截断或仅输出思考）
            return ''

        # 情况2：响应被截断，只有 <think> 没有 </think>
        # 尝试从 <think> 块之外找 JSON（即 <think> 之前的内容）
        if '<think>' in text:
            before_think = text.split('<think>')[0].strip()
            if before_think:
                return before_think
            # <think> 之前没有内容，说明整个响应都是 think 块（被截断）
            # 尝试在 think 内容里找 JSON 结构作为最后手段
            think_content = text.split('<think>', 1)[1]
            first_brace = think_content.find('{')
            last_brace = think_content.rfind('}')
            if first_brace != -1 and last_brace > first_brace:
                return think_content[first_brace:last_brace + 1]
            # 实在找不到，返回空字符串，让上层处理
            return ''

        # 情况3：没有任何 think 标签，直接返回原文
        return text.strip()
    
    def chat(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # 对支持思考模式的模型（如 Qwen3 系列），关闭 think 模式节省 token
        # Ollama 用 payload["think"]=False；vLLM 用 extra_body["enable_thinking"]=False
        try:
            from config import DISABLE_THINKING
            if DISABLE_THINKING:
                ollama_url = OLLAMA_URL  # 从模块级变量获取
                if ollama_url and self.api_url.startswith(ollama_url.rstrip('/')):
                    # Ollama 后端：think 参数直接放在 payload 顶层
                    payload["think"] = False
                else:
                    # vLLM / 其他兼容端点：使用 enable_thinking 扩展字段
                    payload["extra_body"] = {"enable_thinking": False}
        except (ImportError, AttributeError):
            pass
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=1200
            )
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            # 处理 Qwen3 等模型的思考模式输出
            cleaned_content = self._remove_think_tags(content)

            # 若清理后为空，说明 max_tokens 被 think 块耗尽，给出明确提示
            if not cleaned_content:
                if '<think>' in content:
                    print(f"⚠️ Warning: 模型响应仅含 <think> 内容，JSON 输出部分为空。"
                          f"建议增大 max_tokens（当前：{max_tokens}）或在 config.py 中设置 DISABLE_THINKING=True")
                else:
                    print(f"⚠️ Warning: Model returned empty response after cleaning")

            return cleaned_content
        except Exception as e:
            print(f"❌ API调用失败: {e}")
            raise


class OllamaClient(BaseLLMClient):
    """Ollama原生API客户端"""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model
    
    def chat(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
        except Exception as e:
            print(f"Ollama调用失败: {e}")
            raise


class TransformersClient(BaseLLMClient):
    """本地Transformers模型客户端"""
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls, *args, **kwargs):
        # 单例模式，避免重复加载模型
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str, device: str = "cuda"):
        if self._model is not None:
            return  # 已加载，跳过
        
        print(f"正在加载本地模型: {model_path}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.device = device
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            print(f"模型加载完成: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def chat(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        try:
            # 构建对话格式
            messages = [{"role": "user", "content": prompt}]
            
            # 使用chat template（如果支持）
            if hasattr(self._tokenizer, 'apply_chat_template'):
                text = self._tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                text = prompt
            
            inputs = self._tokenizer(text, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # 解码输出（去除输入部分）
            response = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            return response.strip()
        except Exception as e:
            print(f"本地模型推理失败: {e}")
            raise


def get_llm_client() -> BaseLLMClient:
    """
    根据配置获取LLM客户端
    
    Returns:
        LLM客户端实例
    """
    if LLM_BACKEND == "api":
        return OpenAICompatibleClient(API_URL, API_KEY, MODEL_NAME)
    
    elif LLM_BACKEND == "ollama":
        # Ollama也支持OpenAI兼容模式
        return OpenAICompatibleClient(
            f"{OLLAMA_URL}/v1", 
            "ollama",  # Ollama不需要真正的API key
            OLLAMA_MODEL
        )
    
    elif LLM_BACKEND == "vllm":
        return OpenAICompatibleClient(VLLM_URL, "vllm", VLLM_MODEL)
    
    elif LLM_BACKEND == "transformers":
        return TransformersClient(LOCAL_MODEL_PATH, DEVICE)
    
    else:
        raise ValueError(f"不支持的后端类型: {LLM_BACKEND}")


# 全局客户端实例（懒加载）
_global_client: Optional[BaseLLMClient] = None

def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """
    统一的LLM调用接口
    
    Args:
        prompt: 输入提示词
        temperature: 温度参数
        max_tokens: 最大生成token数
        
    Returns:
        模型响应文本
    """
    global _global_client
    if _global_client is None:
        _global_client = get_llm_client()
    
    return _global_client.chat(prompt, temperature, max_tokens)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 测试调用
    test_prompt = "请简要介绍一下什么是机器学习？"
    print(f"测试提示词: {test_prompt}")
    print(f"使用后端: {LLM_BACKEND}")
    print("-" * 50)
    
    response = call_llm(test_prompt)
    print(f"响应: {response}")
