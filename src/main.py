import os
from dotenv import load_dotenv
from openai import OpenAI
from agent.llm_wrapper import LLMWrapper
from agent.planning import Planner
from agent.memory import Memory
from tools.simple_tool import SimpleCalculatorTool

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"]
)

llm = LLMWrapper(client, model="moonshotai/Kimi-K2-Instruct-0905")
planner = Planner(llm)
memory = Memory()
tools = {"SimpleCalculatorTool": SimpleCalculatorTool()}

while True:
    user_input = input("User: ")

    # Add user input to memory first
    memory.add(role="user", content=user_input)

    # Pass full conversation history to planner
    plan = planner.plan(user_input, memory.get())

    if plan["action"] == "tool" and plan["tool_name"] in tools:
        tool = tools[plan["tool_name"]]
        tool_result = tool.run(**plan["arguments"])
        # Store tool result as assistant message
        memory.add(role="assistant", content=f"Tool {plan['tool_name']} result: {tool_result}")
        print(f"Agent (tool result): {tool_result}")
    else:
        # LLM response sees full conversation
        llm_response = llm.generate(memory.get())
        memory.add(role="assistant", content=llm_response)
        print(f"Agent: {llm_response}")
