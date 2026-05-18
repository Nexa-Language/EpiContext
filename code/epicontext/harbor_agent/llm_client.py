"""LLM 客户端: 调用 OpenAI-compatible API。"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple


class LLMClient:
    """真实 LLM API 客户端。"""

    def __init__(self):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        base_url = os.environ.get("OPENAI_API_BASE")
        if not base_url:
            raise ValueError("OPENAI_API_BASE environment variable is required")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = os.environ.get("OPENAI_MODEL_NAME", "mimo-v2.5-pro")
        self.call_count: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512,
             temperature: float = 0.3) -> Tuple[str, int, int]:
        """发送 chat completion 请求。返回 (content, input_tokens, output_tokens)。"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self.call_count += 1
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            return response.choices[0].message.content or "", input_tokens, output_tokens
        except Exception as e:
            return f"Error: {e}", 0, 0
