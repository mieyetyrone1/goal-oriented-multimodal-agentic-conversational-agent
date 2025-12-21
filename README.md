# Goal-Oriented Multimodal Agentic Conversational Agent

This project implements a modular, goal-oriented conversational agent following modern agentic design patterns.
The system is designed to evolve incrementally, starting from a core agent loop and expanding to include reflection, memory management, retrieval-augmented generation (RAG), and Model Context Protocol (MCP).

## Current Status
**Week 1 – Core Agent Loop**
- LLM wrapper using OpenAI-compatible APIs
- LLM-driven planner for tool vs. response decisions
- Tool abstraction and execution
- Conversation memory management

## Architecture (Week 1)
User Input
   ↓
Planner (LLM-driven)
   ↓
Tool Execution (if needed)
   ↓
Memory Update
   ↓
LLM Response

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
