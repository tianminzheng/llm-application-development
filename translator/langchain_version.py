import os

from langchain_core.output_parsers import StrOutputParser

from constants import openai_key
from langchain_openai import OpenAI

from langchain_core.prompts import PromptTemplate

os.environ['OPENAI_API_KEY']=openai_key

text = "你好"
target_language = 'en'

def translatedText():
    template1 = '''以下内容是哪种语言：{text} '''
    language_prompt1 = PromptTemplate(
        input_variables=['text'],
        template=template1
    )

    # Format the language prompt with user input
    language_prompt1.format(text=text)

    template2 = '''将{text}翻译成{target_language} '''
    language_prompt2= PromptTemplate(
        input_variables=['text', 'target_language'],
        template=template2
    )

    # 用用户输入格式化提示词
    language_prompt2.format(text=text, target_language=target_language)

    # 创建OpenAI模型实例
    llm = OpenAI()
    parser = StrOutputParser()

    chain = language_prompt1 | llm | parser

    output_json = chain.invoke({'text': text})
    print(f"Original language: {output_json}")

    chain = language_prompt2 | llm | parser

    # 基于用户输入调用链
    output_json = chain.invoke({'text': text, 'target_language': target_language})

    print(f"Translated to {target_language}: {output_json}")



translatedText()
