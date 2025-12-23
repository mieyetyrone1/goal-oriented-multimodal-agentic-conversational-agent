# Goal-Oriented Multimodal Agentic Conversational Agent

This project implements a modular, goal-oriented conversational agent following modern agentic design patterns.
The system is designed to evolve incrementally, starting from a core agent loop and expanding to include reflection, memory management, retrieval-augmented generation (RAG), and Model Context Protocol (MCP).

## Current Status
**Week 3 – Retrieval-Augmented Generation (RAG)**
- Embedding-based document retrieval using cosine similarity
- Planner-compatible retrieval as an agent tool
- Relevance scoring with configurable similarity thresholds
- Ephemeral context injection (retrieved knowledge is not stored in memory)
- Source-aware responses with deterministic citations
- Clean separation between conversational memory, reflective memory, and retrieved context

Retrieved documents are injected only for the current generation step and are never written to long-term memory, preserving clear memory boundaries and enabling future MCP-based context control.

## Architecture (Week 3)
User Input  
↓  
Planner (LLM-driven)  
↓  
Tool Execution (if needed, including retrieval)  
↓  
Retrieval (Embedding Similarity Search)  
↓  
LLM Response (with grounded context and citations)  
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
