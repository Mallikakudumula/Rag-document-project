"""LLM providers over plain HTTP (no vendor SDK needed). Add a class to support another vendor."""
from dataclasses import dataclass

import requests


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicLLM:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.key, self.model = api_key, model

    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMResponse:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60,
                          headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json={"model": self.model, "max_tokens": max_tokens, "system": system,
                                "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        d = r.json()
        return LLMResponse("".join(b.get("text", "") for b in d["content"]),
                           d["usage"]["input_tokens"], d["usage"]["output_tokens"])


class OpenAICompatLLM:
    """Works with OpenAI, Azure OpenAI (set base URL), Ollama, vLLM, etc."""
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.key, self.base, self.model = api_key, base_url.rstrip("/"), model

    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMResponse:
        r = requests.post(f"{self.base}/chat/completions", timeout=60,
                          headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model, "max_tokens": max_tokens,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}]})
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        return LLMResponse(d["choices"][0]["message"]["content"],
                           u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


def build_llm(s):
    if s.provider == "anthropic":
        return AnthropicLLM(s.anthropic_key, s.model)
    if s.provider == "openai":
        return OpenAICompatLLM(s.openai_key, s.openai_base_url, s.openai_model)
    return None   # mock: answer.py uses the offline extractive answerer
