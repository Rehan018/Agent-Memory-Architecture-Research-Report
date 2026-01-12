import os
from typing import List, Optional
from openai import OpenAI, AzureOpenAI

class LLMClient:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o", api_key: str = None, base_url: str = None):
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif provider == "ollama":
            # Ollama usually runs on localhost:11434 and is compatible with OpenAI client
            self.client = OpenAI(
                base_url=base_url or "http://localhost:11434/v1",
                api_key="ollama" # required but unused
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate_response(self, messages: List[dict], temperature: float = 0.7) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
