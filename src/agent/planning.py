from typing import List, Dict, Any
from agent.llm_wrapper import LLMWrapper
import json

class Planner:
    """
    LLM-driven planner that decides whether to call a tool or respond directly.
    """

    SYSTEM_PROMPT = """
You are a planner for a conversational agent.
Given the user input and conversation history, decide:
1. If the agent should respond directly, set "action": "respond".
2. If a tool should be called, set "action": "tool", and provide "tool_name" and "arguments" as a JSON object.
Return ONLY JSON in the following format:
{
  "action": "respond" | "tool",
  "tool_name": null | string,
  "arguments": {}
}
Available tools:
- SimpleCalculatorTool: arguments = {"expression": "<math expression>"}
- Other tools can be added similarly.
Be concise and valid in JSON format.
"""

    def __init__(self, llm: LLMWrapper):
        self.llm = llm

    def plan(self, user_input: str, memory: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        user_input: latest user message
        memory: list of dicts with "role" and "content"
        """
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(memory)  # Add conversation history
        messages.append({"role": "user", "content": user_input})

        # Ask the LLM for a plan
        response_text = self.llm.generate(messages)

        # Convert LLM JSON output to Python dict
        try:
            plan = json.loads(response_text)
        except json.JSONDecodeError:
            # fallback to default safe action
            plan = {"action": "respond", "tool_name": None, "arguments": {}}

        # Ensure all required keys exist
        for key in ["action", "tool_name", "arguments"]:
            if key not in plan:
                plan[key] = None if key == "tool_name" else {}

        return plan
