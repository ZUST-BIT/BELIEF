"""JSON parsing utilities for LLM responses."""

import json
import re
from typing import Dict, Any, Optional


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from a raw LLM response.
    Handles think tags, code fences, and trailing text.
    """
    if not response:
        return None

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
                try:
                    return json.loads(inner[fb:lb + 1], strict=False)
                except json.JSONDecodeError:
                    pass

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
            try:
                return json.loads(json_str, strict=False)
            except json.JSONDecodeError:
                continue

    # Method 2: last ``` ... ``` block
    if "```" in response:
        parts = response.split("```")
        code_blocks = [parts[k].strip() for k in range(1, len(parts), 2)]
        for block in reversed(code_blocks):
            try:
                return json.loads(block, strict=False)
            except json.JSONDecodeError:
                continue

    # Method 3: scan candidate JSON spans
    spans = _scan_json_spans(response)
    if spans:
        spans_sorted = sorted(spans, key=lambda x: x[1] - x[0], reverse=True)
        for start, end in spans_sorted:
            try:
                return json.loads(response[start:end], strict=False)
            except json.JSONDecodeError:
                continue

    # Method 4: whole response
    try:
        return json.loads(response.strip(), strict=False)
    except json.JSONDecodeError:
        pass

    return None
