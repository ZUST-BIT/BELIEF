"""LangChain wrappers around the existing LLM client."""

from typing import Callable, Any, Optional
from langchain_core.runnables import RunnableLambda

from llm_client import call_llm


def build_llm_chain(
    prompt_builder: Callable[[Any], str],
    temperature: float,
    max_tokens: int,
    caller: Optional[Callable[[str], str]] = None,
):
    """
    Build a lightweight LCEL chain: input -> prompt -> LLM response.
    """
    if caller is None:
        def _caller(prompt: str) -> str:
            return call_llm(prompt, temperature=temperature, max_tokens=max_tokens)
    else:
        _caller = caller

    return RunnableLambda(prompt_builder) | RunnableLambda(_caller)
