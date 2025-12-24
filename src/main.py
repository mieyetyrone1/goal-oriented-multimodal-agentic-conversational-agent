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

print("Agent is running. Type 'exit' to quit.\n")

# Agent loop

while True:
    user_input = input("User: ").strip()
    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit"}:
        break
    
    # Advance MCP context lifecycle
    context_controller.step()

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

    if retrieval_context:
        context_controller.add(
            ContextPacket(
                type="retrieved_knowledge",
                content=retrieval_context,
                source="retriever",
                ttl=1,
                priority=2
            )
        )

    # Reflection (read-only injection)
    latest_reflection = memory.get_latest(role="reflection")
    if latest_reflection:
        context_controller.add(
            ContextPacket(
                type="reflection",
                content=latest_reflection,
                source="reflection",
                ttl=5,
                priority=0
            )
        )

    # Conversation Memory
    context_controller.add(
        ContextPacket(
            type="conversation",
            content=memory.get_llm_messages(
                roles=["user", "assistant"],
                last_n=CONTEXT_WINDOW_SIZE
            ),
            source="memory",
            ttl=-1,
            priority=1
        )
    )

    # LLM Invocation
    llm_messages = context_controller.build_messages()
    llm_messages.append({"role": "user", "content": user_input})

    llm_response = llm.generate(llm_messages)

    print(f"Agent: {llm_response}")

    # Store Memory
    memory.add(role="user", content=user_input)
    memory.add(role="assistant", content=llm_response)

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
