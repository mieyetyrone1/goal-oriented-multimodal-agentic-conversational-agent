import os
from dotenv import load_dotenv
from openai import OpenAI
from agent.llm_wrapper import LLMWrapper
from agent.planning import Planner
from agent.memory import Memory
from agent.reflection import Reflection
from tools.simple_tool import SimpleCalculatorTool
from agent.retrieval.embedding_model import EmbeddingModel
from agent.retrieval.retrieval_helper import build_retrieval_context
from agent.context.controller import ContextController
from agent.context.packet import ContextPacket
from speech.audio_controller import AudioController
from speech.mic_capture import record_until_enter
from speech.stt import WhisperSTT
from speech.tts import WindowsTTS


REFLECTION_INTERVAL = 4  # reflection generation period
CONTEXT_WINDOW_SIZE = 10  # how many messages to include in reflection
TOP_K_RETRIEVAL = 3
RETRIEVAL_SCORE_THRESHOLD = 0.6


load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"]
)

llm = LLMWrapper(
    client=client,
    model="moonshotai/Kimi-K2-Instruct-0905",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)

planner = Planner(llm)
memory = Memory()
reflection = Reflection(llm, context_window_size=CONTEXT_WINDOW_SIZE)
embedding_model = EmbeddingModel()
tools = {"SimpleCalculatorTool": SimpleCalculatorTool()}

context_controller = ContextController()

# Temporary document store
DOCUMENTS = [
    {"id": "supervised", "content": "Supervised learning uses labeled data where each input has a known output."},
    {"id": "unsupervised", "content": "Unsupervised learning finds patterns in unlabeled data."},
    {"id": "reinforcement", "content": "Reinforcement learning trains agents using rewards and penalties."},
]

# Initialize speech
audio = AudioController()
stt = WhisperSTT(model_size="base")
tts = WindowsTTS(rate=150)

print("Agent is running. Press ENTER to talk, type 'exit' to quit.\n")

# Agent loop

while True:
    # Input mode selection
    mode = input("\nChoose input mode ([A]udio / [T]ext, type 'exit' to quit): ").strip().lower()
    if mode in {"exit", "quit"}:
        break

    # Audio input
    if mode in {"a", "audio"}:
        print("Press ENTER to start talking...")
        input()  # Wait for push-to-talk
        audio.begin_listening()
        audio_data = record_until_enter()
        audio.end_listening()
        user_input = stt.transcribe(audio_data)
        print(f"User (transcribed): {user_input}")
    # Text input
    elif mode in {"t", "text"}:
        user_input = input("User: ").strip()
        if not user_input:
            continue
    else:
        print("Invalid input mode. Please choose 'A' or 'T'.")
        continue

    # Record audio state in MCP
    audio_packet = ContextPacket(
        type="conversation",
        content=[],
        source="audio",
        ttl=1,  # lasts 1 turn
        priority=10,
        metadata={"audio_state": audio.state}
    )
    context_controller.add(audio_packet)

    # Planning
    plan = planner.plan(
        user_input=user_input,
        memory=memory.get_llm_messages(roles=["user", "assistant"])
    )
    # Tool Execution
    if plan["action"] == "tool" and plan["tool_name"] in tools:
        tool = tools[plan["tool_name"]]
        tool_result = tool.run(**plan["arguments"])

        memory.add(role="user", content=user_input)
        # Store as assistant message (NOT role=tool)
        memory.add(
            role="assistant",
            content=f"Tool {plan['tool_name']} result: {tool_result}"
        )

        print(f"Agent (tool result): {tool_result}")
        continue
    
    # RAG Retrieval
    retrieval_context = build_retrieval_context(
        query=user_input,
        documents=DOCUMENTS,
        embedding_model=embedding_model,
        top_k=TOP_K_RETRIEVAL,
        score_threshold=RETRIEVAL_SCORE_THRESHOLD,
    )
    print("[DEBUG] Retrieval context:\n", retrieval_context)

    # LLM Response
    llm_messages = []

    if retrieval_context:
        llm_messages.append({"role": "system", "content": retrieval_context})

    llm_messages.extend(
        memory.get_llm_messages(
            roles=["user", "assistant"],
            last_n=CONTEXT_WINDOW_SIZE
        )
    )

    llm_messages.append({"role": "user", "content": user_input})

    llm_response = llm.generate(llm_messages)
    memory.add(role="user", content=user_input)
    memory.add(role="assistant", content=llm_response)

    # Record audio state before speaking
    audio_packet = ContextPacket(
        type="conversation",
        content=[],
        source="audio",
        ttl=1,
        priority=10,
        metadata={"audio_state": audio.state}  # should be SPEAKING
    )
    context_controller.add(audio_packet)

    # Speech Output (TTS)
    audio.begin_speaking()
    tts.speak(llm_response)
    audio.end_speaking()

    # Reflection Generation
    if len(memory.get()) % REFLECTION_INTERVAL == 0:
        summary = reflection.reflect(
            memory.get_llm_messages(
                roles=["user", "assistant"],
                last_n=CONTEXT_WINDOW_SIZE
            )
        )
        memory.add(role="reflection", content=summary)
        print(f"[DEBUG] Reflection Summary:\n{summary}\n")

    # Step MCP packets
    context_controller.step()
    print("[MCP] Active packets:", context_controller.dump())