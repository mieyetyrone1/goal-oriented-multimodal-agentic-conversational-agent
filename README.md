# Goal-Oriented Multimodal Agentic Conversational Agent

This project implements a modular, goal-oriented conversational agent following modern agentic design patterns.
The system is designed to evolve incrementally, starting from a core agent loop and expanding to include reflection, memory management, retrieval-augmented generation (RAG), and Model Context Protocol (MCP).

## Current Status
**Week 2 – Reflection & Memory Management**
- Periodic reflection using LLM-generated summaries
- Configurable context window and reflection frequency
- Separation of conversational memory and reflective state
- Foundation for long-horizon planning and RAG

## Architecture (Week 2)
User Input
   ↓
Planner (LLM-driven)
   ↓
Tool Execution (if needed)
   ↓
LLM Response
   ↓
Memory Update
   ↓
Reflection (periodic)

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
