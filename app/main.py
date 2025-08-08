import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from graphrag_sdk.source import URL
from graphrag_sdk import KnowledgeGraph, Ontology
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk.model_config import KnowledgeGraphModelConfig

FALKOR_HOST = os.getenv("FALKOR_HOST", "localhost")
FALKOR_PORT = int(os.getenv("FALKOR_PORT", "6379"))
MODEL_NAME  = os.getenv("LITELLM_MODEL", "openai/gpt-4.1")

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"ok": True}

class BuildRequest(BaseModel):
    urls: list[str]

@app.post("/build")
def build(req: BuildRequest):
    # 1) pick model via LiteLLM
    model = LiteModel(model_name=MODEL_NAME)

    # 2) sources and ontology
    sources = [URL(u) for u in req.urls]
    ontology = Ontology.from_sources(sources=sources, model=model)

    # 3) spin up KG bound to FalkorDB in Render’s private network
    kg = KnowledgeGraph(
        name="default",
        model_config=KnowledgeGraphModelConfig.with_model(model),
        ontology=ontology,
        host=FALKOR_HOST,
        port=FALKOR_PORT,
    )

    kg.process_sources(sources)
    return {"status": "built", "nodes": len(kg.graph.nodes()) if hasattr(kg, "graph") else None}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    model = LiteModel(model_name=MODEL_NAME)
    # In a real app you'd persist ontology config; here we assume we've already built the KG named "default"
    kg = KnowledgeGraph(
        name="default",
        model_config=KnowledgeGraphModelConfig.with_model(model),
        ontology=None,  # the SDK attaches to the existing KG inside FalkorDB by name
        host=FALKOR_HOST,
        port=FALKOR_PORT,
    )
    session = kg.chat_session()
    answer = session.send_message(req.message)
    return {"answer": answer}
