from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o")

from llama_index.core.tools import QueryEngineTool

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)
from llama_index.core import SQLDatabase

engine = create_engine("sqlite:///:memory:", future=True)
metadata_obj = MetaData()
# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)

metadata_obj.create_all(engine)

from sqlalchemy import insert

rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {"city_name": "Berlin", "population": 3645000, "country": "Germany"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

from llama_index.core.query_engine import NLSQLTableQueryEngine

sql_database = SQLDatabase(engine, include_tables=["city_stats"])
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database, tables=["city_stats"], verbose=True, llm=llm
)
sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    description=(
        "Useful for translating a natural language query into a SQL query over"
        " a table containing: city_stats, containing the population/country of"
        " each city"
    ),
)


from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex

cities = ["Toronto", "Berlin", "Tokyo"]
wiki_docs = WikipediaReader().load_data(pages=cities)

# build a separate vector index per city
# You could also choose to define a single vector index across all docs, and annotate each chunk by metadata
vector_tools = []
for city, wiki_doc in zip(cities, wiki_docs):
    vector_index = VectorStoreIndex.from_documents([wiki_doc])
    vector_query_engine = vector_index.as_query_engine()
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=f"Useful for answering semantic questions about {city}",
    )
    vector_tools.append(vector_tool)


from typing import Dict, Any, List, Tuple, Optional
from llama_index.core.tools import QueryEngineTool
from llama_index.core.program import FunctionCallingProgram
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core import ChatPromptTemplate
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.bridge.pydantic import Field, BaseModel


from llama_index.core.llms import ChatMessage, MessageRole

DEFAULT_PROMPT_STR = """
Given previous question/response pairs, please determine if an error has occurred in the response, and suggest \
    a modified question that will not trigger the error.

Examples of modified questions:
- The question itself is modified to elicit a non-erroneous response
- The question is augmented with context that will help the downstream system better answer the question.
- The question is augmented with examples of negative responses, or other negative questions.

An error means that either an exception has triggered, or the response is completely irrelevant to the question.

Please return the evaluation of the response in the following JSON format.

"""


def get_chat_prompt_template(
    system_prompt: str, current_reasoning: Tuple[str, str]
) -> ChatPromptTemplate:
    system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
    messages = [system_msg]
    for raw_msg in current_reasoning:
        if raw_msg[0] == "user":
            messages.append(
                ChatMessage(role=MessageRole.USER, content=raw_msg[1])
            )
        else:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=raw_msg[1])
            )
    return ChatPromptTemplate(message_templates=messages)


class ResponseEval(BaseModel):
    """Evaluation of whether the response has an error."""

    has_error: bool = Field(
        ..., description="Whether the response has an error."
    )
    new_question: str = Field(..., description="The suggested new question.")
    explanation: str = Field(
        ...,
        description=(
            "The explanation for the error as well as for the new question."
            "Can include the direct stack trace as well."
        ),
    )


from llama_index.core.bridge.pydantic import PrivateAttr


def retry_agent_fn(state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Retry agent.

    Runs a single step.

    Returns:
        Tuple of (agent_response, is_done)

    """
    task, router_query_engine = state["__task__"], state["router_query_engine"]
    llm, prompt_str = state["llm"], state["prompt_str"]
    verbose = state.get("verbose", False)

    if "new_input" not in state:
        new_input = task.input
    else:
        new_input = state["new_input"]

    # first run router query engine
    response = router_query_engine.query(new_input)

    # append to current reasoning
    state["current_reasoning"].extend(
        [("user", new_input), ("assistant", str(response))]
    )

    # Then, check for errors
    # dynamically create pydantic program for structured output extraction based on template
    chat_prompt_tmpl = get_chat_prompt_template(
        prompt_str, state["current_reasoning"]
    )
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=ResponseEval,
        prompt=chat_prompt_tmpl,
        llm=llm,
    )
    # run program, look at the result
    response_eval = llm_program(
        query_str=new_input, response_str=str(response)
    )
    if not response_eval.has_error:
        is_done = True
    else:
        is_done = False
    state["new_input"] = response_eval.new_question

    if verbose:
        print(f"> Question: {new_input}")
        print(f"> Response: {response}")
        print(f"> Response eval: {response_eval.dict()}")

    # set output
    state["__output__"] = str(response)

    # return response
    return state, is_done

from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FnAgentWorker

llm = OpenAI(model="gpt-4o")
router_query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(llm=llm),
    query_engine_tools=[sql_tool] + vector_tools,
    verbose=True,
)
agent = FnAgentWorker(
    fn=retry_agent_fn,
    initial_state={
        "prompt_str": DEFAULT_PROMPT_STR,
        "llm": llm,
        "router_query_engine": router_query_engine,
        "current_reasoning": [],
        "verbose": True,
    },
).as_agent()


# response = agent.chat("Which countries are each city from?")
# print(str(response))

response = agent.chat(
    "What is the city in Canada, and what are the top modes of transport for that city?"
)
print(str(response))
#
# response = sql_query_engine.query(
#     "What are the top modes of transporation fo the city with the lowest population?"
# )
# print(str(response.metadata["sql_query"]))
# print(str(response))

# response = agent.chat("What are the sports teams of each city in Asia?")
# print(str(response))






