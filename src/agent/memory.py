from typing import List, Dict, Optional
from datetime import datetime

class Memory:
    """
    Stores structured conversation history and supports retrieval.
    """
    def __init__(self, max_items: int = 100):
        self.history: List[Dict] = []
        self.max_items = max_items

    def add(self, role: str, content: str, tags: Optional[List[str]] = None):
        """
        Add a new memory item.
        """
        item = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or []
        }
        self.history.append(item)
        # Keep memory within max_items
        if len(self.history) > self.max_items:
            self.history.pop(0)

    def get(self) -> List[Dict]:
        """
        Return full memory history.
        """
        return self.history

    def retrieve(self, keyword: Optional[str] = None, role: Optional[str] = None) -> List[Dict]:
        """
        Retrieve relevant memory items by keyword or role.
        """
        results = self.history
        if keyword:
            results = [item for item in results if keyword.lower() in item["content"].lower()]
        if role:
            results = [item for item in results if item["role"] == role]
        return results

    def get_llm_messages(self, last_n: int = None, roles: list[str] = None) -> List[Dict]:
        """
        Return memory history formatted for LLM (only role and content).
        Can filter by roles and limit to last_n messages.
        Only include roles that are valid for the API: 'user', 'assistant', 'system'.
        """
        safe_roles = ["user", "assistant", "system"]
        if roles is not None:
            safe_roles = [r for r in roles if r in safe_roles]

        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in self.history
            if m["role"] in safe_roles
        ]
        if last_n is not None:
            messages = messages[-last_n:]
        return messages

