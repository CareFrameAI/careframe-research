import asyncio
import json
from fastapi import HTTPException
import google.generativeai as genai

from hypotheses.prompts import generate_hypothesis_build_prompt, generate_hypothesis_extend_prompt, generate_open_exploration_prompt
from llms.client import call_llm_async
from pydantic import BaseModel
from typing import Any, List, Dict

class Node(BaseModel):
    id: str
    name: str
    group: str

class Link(BaseModel):
    source: str
    target: str

class GraphData(BaseModel):
    nodes: List[Node]
    links: List[Link]

class GraphCreateRequest(BaseModel):
    seed: str

class GraphModifyRequest(BaseModel):
    node_name: str
    graph_data: GraphData

class GraphResponse(BaseModel):
    nodes: List[Node]
    links: List[Link]

class GraphHypothesisRequest(BaseModel):
    isolated_ontology: str
    full_ontology: str

class GraphHypothesisResponse(BaseModel):
    hypothesis_based_on_isolated_ontology: Dict
    hypotheses_based_on_full_ontology: List[Dict]

async def create_graph(seed: str):
    print('generating graph for ', seed)

    prompt = generate_open_exploration_prompt(seed)
    try:
        response = await call_llm_async(prompt)
        graph_data = json.loads(response)
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def graph_hypothesis_extend(request: GraphModifyRequest):
    node_name = request.node_name
    graph_data = request.graph_data
    prompt = generate_hypothesis_extend_prompt(graph_data, node_name)
    try:
        response = await call_llm_async(prompt)
        updated_graph_data = json.loads(response)
        return updated_graph_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def graph_hypothesis_build(request: GraphHypothesisRequest):
    isolated_ontology = request.isolated_ontology
    full_ontology = request.full_ontology

    prompt = generate_hypothesis_build_prompt(isolated_ontology, full_ontology)
    try:
        response = await call_llm_async(prompt)
        hypothesis_data = json.loads(response)
        return hypothesis_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    