import os
from dotenv import load_dotenv
from openai import OpenAI
from agent.llm_wrapper import LLMWrapper
from agent.planning import Planner
from agent.memory import Memory
from agent.reflection import Reflection
from tools.simple_tool import SimpleCalculatorTool

REFLECTION_INTERVAL = 3  # reflection generation period
CONTEXT_WINDOW_SIZE = 10  # how many messages to include in reflection

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"]
)

llm = LLMWrapper(client, model="moonshotai/Kimi-K2-Instruct-0905")

planner = Planner(llm)
memory = Memory()
reflection = Reflection(llm, context_window_size=CONTEXT_WINDOW_SIZE)

tools = {"SimpleCalculatorTool": SimpleCalculatorTool()}

while True:
    user_input = input("User: ")
    memory.add(role="user", content=user_input)
    plan = planner.plan(user_input, memory.get_llm_messages(roles=["user","assistant"]))

    if plan["action"] == "tool" and plan["tool_name"] in tools:
        tool = tools[plan["tool_name"]]
        tool_result = tool.run(**plan["arguments"])
        # Store tool result as assistant message
        memory.add(role="assistant", content=f"Tool {plan['tool_name']} result: {tool_result}")
        print(f"Agent (tool result): {tool_result}")
    else:
        # LLM response sees full conversation
        llm_response = llm.generate(memory.get_llm_messages(roles=["user","assistant"]))
        memory.add(role="assistant", content=llm_response)
        print(f"Agent: {llm_response}")
    
    if len(memory.get()) % REFLECTION_INTERVAL == 0:
        summary = reflection.reflect(
            memory.get_llm_messages(last_n=CONTEXT_WINDOW_SIZE)
        )
        if summary:
            memory.add(role="reflection", content=summary)
            print(f"\n--- Reflection Summary ---\n{summary}\n")
