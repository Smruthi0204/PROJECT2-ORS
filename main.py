# main.py
import os
import uvicorn
import fitz
import docx
import requests
import tempfile
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from email import message_from_string
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- CONFIG ---
LLM_API_URL = "http://localhost:1234/v1/completions"
LLM_API_KEY = "none"
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- VECTOR DB INIT ---
client = QdrantClient(":memory:")
COLLECTION = "docs"
DIM = 384  # all-MiniLM-L6-v2 dimension

if COLLECTION not in client.get_collections().collections:
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
    )

# --- FASTAPI INIT ---
app = FastAPI(title="LLM Clause Retrieval", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# --- SCHEMA ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

# --- PARSERS ---
def parse_pdf(path): return "\n".join([p.get_text() for p in fitz.open(path)])
def parse_docx(path): return "\n".join([p.text for p in docx.Document(path).paragraphs])
def parse_email(raw): return "\n".join(part.get_payload(decode=True).decode(errors='ignore')
                                       for part in message_from_string(raw).walk()
                                       if part.get_content_type() == "text/plain")

def fetch_and_parse_document(url: str) -> str:
    r = requests.get(url)
    if r.status_code != 200:
        raise HTTPException(400, "Failed to fetch")
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(r.content)
        path = f.name
    if url.endswith(".pdf"): return parse_pdf(path)
    elif url.endswith(".docx"): return parse_docx(path)
    elif url.endswith(".eml") or url.endswith(".msg"): return parse_email(r.text)
    else: raise HTTPException(400, "Unsupported file type")

# --- CHUNK + STORE ---
def chunk_text(text, size=300):
    sents, out, cur = text.split(". "), [], ""
    for s in sents:
        if len(cur) + len(s) < size: cur += s + ". "
        else: out.append(cur.strip()); cur = s + ". "
    if cur: out.append(cur.strip())
    return out

def store_chunks(chunks: List[str]):
    vectors = model.encode(chunks).tolist()
    points = [PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]}) for i in range(len(chunks))]
    client.upsert(COLLECTION, points=points)

def search_chunks(query: str, k=5) -> List[str]:
    vec = model.encode(query).tolist()
    res = client.search(collection_name=COLLECTION, query_vector=vec, limit=k)
    return [r.payload["text"] for r in res]

# --- LLM CALL ---
def build_prompt(question, context):
    return f"""
Use the context below to answer the user query clearly and concisely.

Context:
{"".join(context)}

Question:
{question}

Respond only in JSON like:
{{"answer": "<your answer>"}}
"""

async def call_llm(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    body = {
        "model": "mistral",
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2
    }
    async with httpx.AsyncClient() as c:
        res = await c.post(LLM_API_URL, headers=headers, json=body)
        res.raise_for_status()
        return res.json()["choices"][0]["text"]

# --- API ---
@app.post("/api/v1/hackrx/run")
async def run(req: QueryRequest):
    try:
        doc_text = fetch_and_parse_document(req.documents)
        chunks = chunk_text(doc_text)
        store_chunks(chunks)

        answers = []
        for q in req.questions:
            ctx = search_chunks(q)
            prompt = build_prompt(q, ctx)
            result = await call_llm(prompt)
            try:
                answers.append(json.loads(result)["answer"])
            except:
                answers.append("Could not parse answer. Raw: " + result)

        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- LOCAL RUN ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
