
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

from llama_index.core.tools import BaseTool, FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)


llm = OpenAI(model="gpt-4")
agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool], llm=llm, verbose=True
)

# response = agent.chat("What is (121 * 3) + 42?")
# print(str(response))


response = agent.stream_chat(
    "What is 121 * 2? Once you have the answer, use that number to write a"
    " story about a group of mice."
)

response_gen = response.response_gen

for token in response_gen:
    print(token, end="")