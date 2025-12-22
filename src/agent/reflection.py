from typing import List, Dict
from agent.llm_wrapper import LLMWrapper

class Reflection:
    """
    Generates reflection summaries from recent conversation history.
    """
    SYSTEM_PROMPT = """
        You are an AI assistant tasked with summarizing the conversation so far.
        Provide concise insights, patterns, or points for improvement.
    """
    def __init__(self, llm: LLMWrapper, context_window_size: int = 10):
        self.llm = llm
        self.CONTEXT_WINDOW_SIZE = context_window_size

    def reflect(self, memory: List[Dict]) -> str:
        """
        Reflection is implemented as an explicit user-level summarization task
        (not a system-only meta prompt) to ensure compatibility with
        instruction-tuned LLMs across OpenAI-compatible APIs.
        Generate a reflection summary from recent conversation memory.
        memory: List of dicts with keys {"role", "content"}
        """

        if not memory:
            return ""

        # Limit context window
        recent_memory = memory[-self.CONTEXT_WINDOW_SIZE :]

        # Flatten conversation into plain text (more reliable for summarization)
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent_memory
        )

        prompt = (
            "Summarize the following conversation in 2–4 sentences. "
            "Focus on key questions, answers, and any tool usage. "
            "Highlight important concepts or patterns.\n\n"
            f"{conversation_text}"
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        summary = self.llm.generate(messages)

        return summary.strip()
