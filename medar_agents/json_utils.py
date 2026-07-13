"""JSON parsing utilities for LLM responses."""

import json
import re
from typing import Dict, Any, Optional


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from a raw LLM response.
    Handles think tags, code fences, and trailing text.
    """
    if not isinstance(response, str) or not response.strip():
        return None

    def _load_object(candidate: str) -> Optional[Dict[str, Any]]:
        """Parse one JSON candidate and accept objects only."""
        try:
            value = json.loads(candidate, strict=False)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    # Step 0: strip <think>...</think> blocks if present
    if "</think>" in response:
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    elif "<think>" in response:
        before = response.split("<think>")[0].strip()
        if before:
            response = before
        else:
            inner = response.split("<think>", 1)[1]
            fb, lb = inner.find("{"), inner.rfind("}")
            if fb != -1 and lb > fb:
                result = _load_object(inner[fb:lb + 1])
                if result is not None:
                    return result

    if not response:
        return None

    def _scan_json_spans(text: str):
        spans = []
        n = len(text)
        i = 0
        while i < n:
            if text[i] != "{":
                i += 1
                continue
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                ch = text[j]
                if esc:
                    esc = False
                elif ch == "\\" and in_str:
                    esc = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            spans.append((i, j + 1))
                            break
                j += 1
            i += 1
        return spans

    # Method 1: last ```json ... ``` block
    if "```json" in response:
        parts = response.split("```json")
        for chunk in reversed(parts[1:]):
            json_str = chunk.split("```")[0].strip()
            result = _load_object(json_str)
            if result is not None:
                return result

    # Method 2: last ``` ... ``` block
    if "```" in response:
        parts = response.split("```")
        code_blocks = [parts[k].strip() for k in range(1, len(parts), 2)]
        for block in reversed(code_blocks):
            result = _load_object(block)
            if result is not None:
                return result

    # Method 3: scan candidate JSON spans
    spans = _scan_json_spans(response)
    if spans:
        spans_sorted = sorted(spans, key=lambda x: x[1] - x[0], reverse=True)
        for start, end in spans_sorted:
            result = _load_object(response[start:end])
            if result is not None:
                return result

    # Method 4: whole response
    return _load_object(response.strip())
