import time
import together
from together import Together

api = "a1c3595fc04659567466b8324c0948fe979577f2bec891f7924ca33b5d2363b0" # see readme about how to get api

client = Together(api_key=api)

reference_models = [
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "databricks/dbrx-instruct",
]

aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

aggregator_system_prompt = """你已经从各种开源模型获得了针对最新用户查询的一组响应。你的任务是将这些响应综合成一个单一的、高质量的回复。至关重要的是要批判性地评估这些响应中提供的信息，认识到其中一些信息可能是有偏见或不正确的。你的回复不应简单地复制给定的答案，而应提供一个精炼、准确和全面的回复来回应指令。确保你的回复结构良好、连贯，并遵循准确性和可靠性的最高标准。

模型的响应：
"""

layers = 3


def get_final_system_prompt(system_prompt, results):
    """构建一个针对2层及以上的系统提示词，其中整合了来自先前模型的响应结果。"""
    return (
        system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )


def run_llm(model, user_prompt, prev_response=None):
    """运行一个单一的LLM调用，并考虑之前的响应结果和限流机制。"""
    messages = (
        [
            {
                "role": "system",
                "content": get_final_system_prompt(
                    aggregator_system_prompt, prev_response
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        if prev_response
        else [{"role": "user", "content": user_prompt}]
    )
    for sleep_time in [1, 2, 4]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=512,
            )
            print("Model: ", model)

            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except together.error.RateLimitError as e:
            print(e)
            time.sleep(sleep_time)
    return None


def moa_generate(user_prompt):
    """运行MOA流程的主循环，并返回最终结果。"""

    # 第一步：针对各个提议者模型，对用户输入执行调用
    results = [run_llm(model, user_prompt) for model in reference_models]

    # 第二步：基于MoA架构的层级，再次对用户输入执行模型调用，并整合第一步中的模型响应结果作为下一个模型的输入
    for _ in range(1, layers - 1):
        results = [run_llm(model, user_prompt, prev_response=results) for model in reference_models]

    # 第三步：针对聚合者模型执行调用，并整合第二步中的模型响应结果作为最终输入
    print("Model: ", aggregator_model)
    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {
                "role": "system",
                "content": get_final_system_prompt(aggregator_system_prompt, results),
            },
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    final_response = ""
    for chunk in finalStream:
        final_response += chunk.choices[0].delta.content or ""
    # print(final_response)
    return final_response


from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"


def get_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant.",
            ),
            ("assistant", "{assistant}"),
            ("user", "{user}")
        ]
    )
    return prompt

# Example usage:
if __name__ == "__main__":
    user_prompt = "列举杭州的3个著名景点?"
    final_result = moa_generate(user_prompt)
    print(final_result)

