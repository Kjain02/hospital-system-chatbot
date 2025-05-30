import os
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import initialize_agent, AgentType, create_openai_functions_agent, create_tool_calling_agent
from langchain.tools import tool, Tool
# from langchain.agents.output_parsers import  ReActJsonSingleInputOutputParser
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
from chains.hospital_cypher_chain import hospital_cypher_chain
from chains.hospital_review_chain import reviews_vector_chain

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# class ExperienceInput(BaseModel):
#     query: str = Field(description="Pass the user query unchanged")

# class GraphInput(BaseModel):
#     query: str = Field(description="Pass the user query unchanged")


# @tool("experiences", return_direct=True, args_schema=ExperienceInput)
# def Experiences(query: str) -> str:
#     """Useful when you need to answer questions
#         about patient experiences, feelings, issues, complaints, patient's 
#         views about the hospital or any other qualitative
#         question that could be answered about a patient using semantic
#         search. Not useful for answering objective questions that involve
#         counting, percentages, aggregations, or listing facts. Use the
#         entire prompt as input to the tool. For example, if the prompt is
#         "Are patients satisfied with their care?", the input should be
#         "Are patients satisfied with their care?".
#         """
#     response = reviews_vector_chain.invoke(query)
    
#     return response.get("result")

# @tool("graph", return_direct=True, args_schema=GraphInput)
# def Graph(query: str) -> str:
#     """Useful for answering questions about patients details,
#         physicians, hospitals, insurance payers, patient review
#         statistics, and hospital visit details. Use the entire prompt as
#         input to the tool. For example, if the prompt is "How many visits
#         have there been?", the input should be "How many visits have
#         there been?".
#         """
#     response = hospital_cypher_chain.invoke(query)
#     return response.get("result")

def safe_cypher_invoke(chain):
    def wrapper(input_str):
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string.")
        return chain.invoke({"query": input_str})
    return wrapper

def safe_review_invoke(chain):
    def wrapper(input_str):
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string.")
        return chain.invoke({"question": input_str})
    return wrapper


Experiences_Tool = Tool(
        name="Experience",
        func=safe_review_invoke(reviews_vector_chain),
        description="""Use this tool to answer subjective or qualitative questions from patient reviews 
        (e.g. emotions, satisfaction, experiences with a specific doctor or hospital). Not useful for answering
        objective questions that involve counting, percentages, aggregations, or listing facts. 
        Examples: 
        - "How did patients feel about Dr. Ashley Le?"
        - "What complaints were raised at St. Mary's Hospital?"
        Use the entire prompt as input to the tool. For example, if the prompt is
        "Are patients satisfied with their care?", the input should be
        "Are patients satisfied with their care?".
        """
    )

Graph_Tool = Tool(
        name="Graph",
        func=safe_cypher_invoke(hospital_cypher_chain),
        description="""Useful for answering questions about patients details,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For example, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?".
        """
    )


tools = [Experiences_Tool, Graph_Tool]
# tools = [Experiences, Graph]

chat_model = ChatOllama(model="llama3", temperature=0)

# llm = ChatOpenAI(
#     api_key="ollama",
#     model="llama3",
#     base_url="http://localhost:11434/v1",
# )

# llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)



# hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")
memory = ConversationBufferMemory(memory_key="chat_history") 
hospital_rag_agent = initialize_agent(
    llm=chat_model,
    tools=tools,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    
    )




# hospital_rag_agent_executor = AgentExecutor(
#     agent=hospital_rag_agent,
#     tools=tools,
#     return_intermediate_steps=True,
#     verbose=True
# )