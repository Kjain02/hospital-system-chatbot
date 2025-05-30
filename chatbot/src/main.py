from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from agents.hospital_rag_agent import hospital_rag_agent
from utils.async_utils import async_retry

app = FastAPI(
    title = "Hospital Chatbot",
    description="Endpoints for hospital system graph chatbot"
)

class HospitalQueryInput(BaseModel):
    text: str

class HospitalQueryOutput(BaseModel):
    output: str
    intermediate_steps: List[str]

@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agent if a tool fails to run. When there
    are intermittent connection issues to external APIs. Like to OpenAI or Database
    """
    return await hospital_rag_agent.ainvoke({"input": query})

@app.get("/")
async def get_status():
    return {"status": "running"}

@app.post("/hospital-rag-agent")
async def query_hospital_agent(
    query: HospitalQueryInput
) -> HospitalQueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    return HospitalQueryOutput(
        output=query_response.get("output", ""),
        intermediate_steps=[str(s) for s in query_response.get("intermediate_steps", [])]
    )