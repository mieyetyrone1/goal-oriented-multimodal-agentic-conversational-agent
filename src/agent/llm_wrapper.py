from openai import OpenAI
from typing import List, Dict

class LLMWrapper:
    """
    A wrapper for OpenAI-compatible API.
    """

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: [{"role": "system|user|assistant", "content": "..."}]
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        # return response.choices[0].message["content"]
        # HuggingFace ChatCompletion returns ChatCompletionMessage object
        return response.choices[0].message.content
