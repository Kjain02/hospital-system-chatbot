import os
from dotenv import load_dotenv
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

load_dotenv(override=True)

MODEL_API_URL = os.getenv("MODEL_API_URL")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")



neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding_node_property="embedding",
    embedding=HuggingFaceEmbeddings(),
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="reviews",
    node_label="Review",
    text_node_properties=[
        "physician_name",
        "patient_name",
        "text",
        "hospital_name",
    ]
)

review_template = """Your job is to use patient
reviews to answer questions about their experience at a hospital. Use
the following context to answer questions. Be as detailed as possible, but
don't make up any information that's not from the context. If you don't know
an answer, say "I don't know" and nothing more.

Always format your answer as:

Final Answer: <your answer here>

Context:
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=review_template)
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [review_system_prompt, review_human_prompt]

review_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

llm = ChatOllama(model="llama3", temperature=0, top_p=1, top_k=1, verbose=True)
# llm = HuggingFaceEndpoint(
#     endpoint_url=MODEL_API_URL,
#     task="text-generation",
#     max_new_tokens=2048,
#     top_k=10,
#     temperature=0,
#     repetition_penalty=1.03,
#     huggingfacehub_api_token=HF_TOKEN
# )


# retriever = MultiQueryRetriever.from_llm(
#     llm=llm,
#     retriever=neo4j_vector_index.as_retriever(k=12),
#     include_original=True
#     )

retriever = neo4j_vector_index.as_retriever(k=8)

def validate_context(inputs):
    if not inputs["context"]:
        return {"context": "No relevant reviews found.", "question": inputs["question"]}
    return inputs

reviews_vector_chain = (
    {
        "context": RunnableLambda(lambda x: retriever.invoke(x["question"])), 
        "question": RunnablePassthrough()
    }
    | RunnableLambda(validate_context)
    | review_prompt
    | llm
    )

# print("Welcome to the Hospital Review Chatbot!")
# while True:
#     user_query = input("Ask query: ")
#     # print("You asked:", user_query)
#     if user_query.lower() in ["bye", "thank you"]:
#         break
#     response = reviews_vector_chain.invoke({"question": user_query})
#     ans = response.content
#     print(ans)
# reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt



# questions = [
#    "What is the current wait time at Wallace-Hamilton hospital?",
#    "Which hospital has the shortest wait time?",
#    "At which hospitals are patients complaining about billing and insurance issues?",
#    "What is the average duration in days for emergency visits?",
#    "What are patients saying about the nursing staff at Castaneda-Hardy?",
#    "What was the total billing amount charged to each payer for 2023?",
#    "What is the average billing amount for Medicaid visits?",
#    "How many patients has Dr. Ryan Brown treated?",
#    "Which physician has the lowest average visit duration in days?",
#    "How many visits are open and what is their average duration in days?",
#    "Have any patients complained about noise?",
#    "How much was billed for patient 789's stay?",
#    "Which physician has billed the most to cigna?",
#    "Which state had the largest percent increase in Medicaid visits from 2022 to 2023?",
# ]