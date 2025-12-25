from dataclasses import dataclass, field
from typing import Any, Optional, Dict

@dataclass
class ContextPacket:
    """
    Represents a unit of context with explicit lifetime and source.
    """
    type: str                # e.g. "conversation", "retrieved_knowledge", "reflection"
    content: Any             # str or structured data
    source: str              # e.g. "memory", "retriever", "reflection"
    ttl: int                 # -1 = persistent, >0 = decrement each turn
    priority: int = 0        # higher = appears earlier in prompt
    metadata: Optional[Dict] = field(default_factory=dict)  # <-- new field

    def is_expired(self) -> bool:
        return self.ttl == 0

    def step(self):
        if self.ttl > 0:
            self.ttl -= 1
