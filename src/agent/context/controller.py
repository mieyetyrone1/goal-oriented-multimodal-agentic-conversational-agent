from typing import List
from agent.context.packet import ContextPacket


class ContextController:
    """
    Manages context packets and assembles LLM-ready messages.
    """

    def __init__(self):
        self._packets: List[ContextPacket] = []

    def add(self, packet: ContextPacket):
        self._packets.append(packet)

    def step(self):
        """
        Advance context lifecycle by one turn.
        Decrements TTLs and removes expired packets.
        """
        for packet in self._packets:
            packet.step()

        self._packets = [
            packet for packet in self._packets if not packet.is_expired()
        ]

    def build_messages(self) -> List[dict]:
        """
        Assemble messages for LLM consumption.
        """
        messages = []

        # Sort packets by priority (descending)
        packets = sorted(self._packets, key=lambda p: p.priority, reverse=True)

        for packet in packets:
            rendered = self._render_packet(packet)
            if rendered:
                messages.extend(rendered)

        return messages

    def _render_packet(self, packet: ContextPacket) -> List[dict]:
        """
        Convert a ContextPacket into LLM messages.
        """
        if packet.type in {"retrieved_knowledge", "reflection"}:
            return [{
                "role": "system",
                "content": packet.content
            }]

        if packet.type == "conversation":
            # content is already a list of messages
            return packet.content

        # Unknown packet type → ignore safely
        return []

    def dump(self) -> List[dict]:
        """
        Debug utility: inspect active context packets.
        """
        return [
            {
                "type": p.type,
                "source": p.source,
                "ttl": p.ttl,
                "priority": p.priority
            }
            for p in self._packets
        ]
