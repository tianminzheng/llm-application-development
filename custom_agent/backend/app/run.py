
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from bot import *

input_files = ["transformers.pdf"]
chatbot = Chat(input_files=input_files)


# response = chatbot.process_query("What is Attention?")
# print(str(response))


# response = chatbot.process_query("How many types of Attention?")
# print(str(response))


response = chatbot.process_query("List the contributors to this paper written on the first page, please!?")
print(str(response))
