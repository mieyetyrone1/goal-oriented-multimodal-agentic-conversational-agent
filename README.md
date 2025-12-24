# Goal-Oriented Multimodal Agentic Conversational Agent

This project implements a modular, goal-oriented conversational agent following modern agentic design patterns.
The system is designed to evolve incrementally, starting from a core agent loop and expanding to include reflection, memory management, retrieval-augmented generation (RAG), and Model Context Protocol (MCP).

## Current Status
**Week 4 – Model Context Protocol (MCP)**
- Explicit context packets with type, source, priority, and lifetime (TTL)
- Centralized context controller for assembling LLM prompts
- Clear separation between:
  - Conversational memory
  - Reflective memory
  - Retrieved knowledge (RAG)
- Ephemeral retrieval context (no memory pollution)
- Reflection-aware context injection
- Deterministic, inspectable prompt construction
- Planner-compatible context orchestration

MCP serves as the control plane for all context entering the model, enabling predictable behavior, safe memory boundaries, and future extensibility.

## Architecture (Week 4)
User Input  
↓  
Planner (LLM-driven)  
↓  
Tool Execution (if needed)  
↓  
Retrieval (Embedding Similarity Search)  
↓  
Context Assembly (MCP)  
↓  
LLM Response  
↓  
Memory Update  
↓  
Reflection (periodic)

Context Types Managed by MCP:
- Conversation history (persistent)
- Retrieved knowledge (TTL-scoped)
- Reflection summaries (medium-lived)
- System-level instructions

## Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token
```
## Execution

Run the agent from the project root
py ./src/main.py
