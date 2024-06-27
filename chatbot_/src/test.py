import dotenv
dotenv.load_dotenv()

# from agents.hospital_rag_agent import hospital_rag_agent


# while True:
#     user_query = input("Ask query: ")
#     if user_query.lower() in ["bye", "thank you"]:
#         break
#     response = hospital_rag_agent.invoke({"input": user_query})
#     ans = response["output"]
#     print(ans)
from chains.hospital_review_chain import reviews_vector_chain

response = reviews_vector_chain.invoke("At which hospitals are patients are facing problems and complaining about billing and insurance issues?")
print(response)