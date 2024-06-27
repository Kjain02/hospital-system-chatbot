import os
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

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
an answer, say you don't know.
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

llm = ChatOllama(model="llama3", temperature=0, verbose=True)
# llm = HuggingFaceEndpoint(
#     endpoint_url=MODEL_API_URL,
#     task="text-generation",
#     max_new_tokens=2048,
#     top_k=10,
#     temperature=0,
#     repetition_penalty=1.03,
#     huggingfacehub_api_token=HF_TOKEN
# )


retriever = MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=neo4j_vector_index.as_retriever(k=12),
    include_original=True
    )

reviews_vector_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | review_prompt
    | llm
    )
# reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt