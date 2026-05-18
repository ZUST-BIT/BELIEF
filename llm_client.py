"""
LLM统一客户端模块
支持多种模型调用方式：
1. OpenAI兼容API（付费API、vLLM、Ollama等）
2. 本地Transformers加载
"""

import requests
from typing import Optional
from abc import ABC, abstractmethod
# ==================== 配置参数 ====================
# 选择使用的后端: "api" | "ollama" | "vllm" | "transformers"
LLM_BACKEND = "ollama"

# API配置
API_URL = "https://api.gptsapi.net/v1"
API_KEY = "sk-..."
MODEL_NAME = "gpt-4o-mini"

# Ollama配置
OLLAMA_URL = "http://172.18.51.166:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# vLLM配置
VLLM_URL = "http://172.18.51.166:8001/v1"
VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 本地Transformers配置
LOCAL_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda"  # "cuda" 或 "cpu"
# ================================================

def get_disable_thinking() -> bool:
    """
    从 config.py 读取 DISABLE_THINKING，默认 False
    """
    try:
        from config import DISABLE_THINKING
        return bool(DISABLE_THINKING)
    except Exception:
        return False


def _get_model_reasoning_level(model_name: str) -> Optional[str]:
    """
    从 config.py 读取指定模型的推理级别（如 low/medium/high）
    """
    try:
        from config import MODEL_REASONING_LEVELS
        if isinstance(MODEL_REASONING_LEVELS, dict):
            level = MODEL_REASONING_LEVELS.get(model_name)
            if level is None:
                return None
            level = str(level).strip()
            return level or None
    except Exception:
        pass
    return None


def _get_model_system_prompt(model_name: str) -> Optional[str]:
    """
    从 config.py 读取指定模型的 system prompt。
    若同时配置了推理级别，会自动附加一行 `Reasoning: <level>`。
    """
    base_prompt = ""
    try:
        from config import MODEL_SYSTEM_PROMPTS
        if isinstance(MODEL_SYSTEM_PROMPTS, dict):
            base_prompt = str(MODEL_SYSTEM_PROMPTS.get(model_name, "") or "").strip()
    except Exception:
        base_prompt = ""

    reasoning_level = _get_model_reasoning_level(model_name)
    if reasoning_level:
        reasoning_line = f"Reasoning: {reasoning_level}"
        if base_prompt:
            return f"{reasoning_line}\n{base_prompt}"
        return reasoning_line

    return base_prompt or None


def _get_model_disable_thinking_override(model_name: str) -> Optional[bool]:
    """
    从 config.py 读取指定模型的 think 开关覆盖。
    返回：True/False 表示覆盖全局配置；None 表示沿用全局 DISABLE_THINKING。
    """
    try:
        from config import MODEL_DISABLE_THINKING_OVERRIDES
        if isinstance(MODEL_DISABLE_THINKING_OVERRIDES, dict) and model_name in MODEL_DISABLE_THINKING_OVERRIDES:
            return bool(MODEL_DISABLE_THINKING_OVERRIDES[model_name])
    except Exception:
        pass
    return None


def get_disable_thinking_for_model(model_name: str) -> bool:
    """
    计算某个模型最终生效的 DISABLE_THINKING。
    优先级：模型覆盖 > 全局 DISABLE_THINKING。
    """
    override = _get_model_disable_thinking_override(model_name)
    if override is not None:
        return override
    return get_disable_thinking()


def _remove_think_tags(text: str) -> str:
    """
    移除 Qwen3 等模型的思考标签 <think>...</think>

    处理三种情况：
    1. 正常闭合的 <think>...</think>
    2. 只有 <think> 没有 </think>（被截断）
    3. 没有 think 标签
    """
    import re

    if not text:
        return text

    if "</think>" in text:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if cleaned:
            return cleaned

        after_think = text.split("</think>")[-1].strip()
        if after_think:
            return after_think

        return ""

    if "<think>" in text:
        before_think = text.split("<think>")[0].strip()
        if before_think:
            return before_think

        think_content = text.split("<think>", 1)[1]
        first_brace = think_content.find("{")
        last_brace = think_content.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            return think_content[first_brace:last_brace + 1].strip()

        return ""

    return text.strip()


class BaseLLMClient(ABC):
    """LLM客户端基类"""

    @abstractmethod
    def chat(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """发送对话请求"""
        pass


class OpenAICompatibleClient(BaseLLMClient):
    """
    OpenAI兼容API客户端
    支持：
    - 付费 OpenAI 兼容 API
    - vLLM OpenAI 兼容接口
    - Ollama 的 OpenAI 兼容接口（如果你想走 /v1）
    """

    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _build_payload(self, prompt: str, temperature: float, max_tokens: int) -> dict:
        """
        构建 OpenAI 兼容接口请求体
        并根据 DISABLE_THINKING 为不同后端附加不同参数
        """
        messages = []
        system_prompt = _get_model_system_prompt(self.model)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        disable_thinking = get_disable_thinking_for_model(self.model)

        if disable_thinking:
            # vLLM：Qwen3 关闭 thinking 的常用方式
            # 注意：这是直接写进 JSON 顶层，不是 extra_body
            if self.api_url.startswith(VLLM_URL.rstrip("/")):
                payload["chat_template_kwargs"] = {
                    "enable_thinking": False
                }

            # Ollama OpenAI兼容接口（如果你使用的是 /v1/chat/completions）
            elif self.api_url.startswith(f"{OLLAMA_URL.rstrip('/')}/v1"):
                payload["think"] = False

        return payload

    def chat(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = self._build_payload(prompt, temperature, max_tokens)

        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=1200
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"].strip()
            cleaned_content = _remove_think_tags(content)

            if not cleaned_content:
                if "<think>" in content:
                    print(
                        f"⚠️ Warning: 模型响应仅含 <think> 内容，JSON/正文为空。"
                        f"建议增大 max_tokens（当前：{max_tokens}）"
                        f"或确认 DISABLE_THINKING=True 是否已生效。"
                    )
                else:
                    print("⚠️ Warning: Model returned empty response after cleaning")

            return cleaned_content

        except requests.HTTPError as e:
            response_text = ""
            if getattr(e, "response", None) is not None:
                response_text = (e.response.text or "")[:800]
            status_code = getattr(getattr(e, "response", None), "status_code", "unknown")
            print(f"❌ API调用失败: HTTP {status_code}. 响应片段: {response_text}")
            raise
        except Exception as e:
            print(f"❌ API调用失败: {e}")
            raise


class OllamaClient(BaseLLMClient):
    """
    Ollama原生API客户端
    走 /api/generate
    """

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        disable_thinking = get_disable_thinking_for_model(self.model)
        system_prompt = _get_model_system_prompt(self.model)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        # Ollama 原生接口支持顶层 think 参数
        if disable_thinking:
            payload["think"] = False

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=1200
            )
            response.raise_for_status()
            result = response.json()

            content = result.get("response", "").strip()
            cleaned_content = _remove_think_tags(content)

            if not cleaned_content:
                if "<think>" in content:
                    print(
                        f"⚠️ Warning: Ollama响应仅含 <think> 内容。"
                        f"建议检查 DISABLE_THINKING=True 是否生效，"
                        f"或适当增大 num_predict（当前：{max_tokens}）"
                    )
                else:
                    print("⚠️ Warning: Ollama returned empty response after cleaning")

            return cleaned_content

        except requests.HTTPError as e:
            response_text = ""
            if getattr(e, "response", None) is not None:
                response_text = (e.response.text or "")[:800]
            status_code = getattr(getattr(e, "response", None), "status_code", "unknown")
            print(f"❌ Ollama调用失败: HTTP {status_code}. 响应片段: {response_text}")
            raise
        except Exception as e:
            print(f"❌ Ollama调用失败: {e}")
            raise


class TransformersClient(BaseLLMClient):
    """本地Transformers模型客户端"""

    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls, *args, **kwargs):
        # 单例模式，避免重复加载模型
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str, device: str = "cuda"):
        if self._model is not None:
            return

        print(f"正在加载本地模型: {model_path}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self._device = device
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
            messages = [{"role": "user", "content": prompt}]

            if hasattr(self._tokenizer, "apply_chat_template"):
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = prompt

            inputs = self._tokenizer(text, return_tensors="pt")
            if self._device == "cuda":
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

            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            return _remove_think_tags(response)

        except Exception as e:
            print(f"本地模型推理失败: {e}")
            raise


def get_llm_client() -> BaseLLMClient:
    """
    根据配置获取LLM客户端
    """
    if LLM_BACKEND == "api":
        return OpenAICompatibleClient(API_URL, API_KEY, MODEL_NAME)

    elif LLM_BACKEND == "ollama":
        # 推荐走 Ollama 原生接口，这样 think=false 更稳
        return OllamaClient(OLLAMA_URL, OLLAMA_MODEL)

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