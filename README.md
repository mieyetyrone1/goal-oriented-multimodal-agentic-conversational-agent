# Goal-Oriented Multimodal Agentic Conversational Agent

This project implements a modular, goal-oriented conversational agent following modern agentic design patterns.
The system is designed to evolve incrementally, starting from a core agent loop and expanding to include reflection, memory management, retrieval-augmented generation (RAG), and Model Context Protocol (MCP).

## Current Status
**Week 5 – Model Context Protocol (MCP)**
- Push-to-talk audio input using Windows microphone
- Whisper STT transcription for audio input
- Windows TTS for agent speech output
- Half-duplex audio management to prevent simultaneous speaking/listening
- MCP integration: audio state recorded in `ContextPacket.metadata` (`IDLE`, `LISTENING`, `SPEAKING`)
- Flexible dual-mode input: user can switch per turn between **Audio** and **Text**
- Existing planner, RAG, memory, reflection, and tool execution remain fully compatible
- Ephemeral audio packets in MCP for context-aware LLM interactions
- Debug-friendly inspection of audio and MCP state

## Architecture (Week 5)
User Input (Audio/Text)
      ↓
  Planner (LLM-driven)
      ↓
  Tool Execution (if needed)
      ↓
  RAG Retrieval + Context Injection
      ↓
  LLM Response
      ↓
  Memory Update
      ↓
  Reflection (periodic)
      ↓
  TTS (if audio mode)

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
