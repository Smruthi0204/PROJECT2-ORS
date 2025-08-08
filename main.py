# main.py
import os
import uvicorn
import fitz  # PyMuPDF
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
import chromadb
from chromadb.utils import embedding_functions

### CONFIGURATION
LLM_API_URL = "http://localhost:1234/v1/completions"
LLM_API_KEY = "none"

### INIT CHROMA DB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

### FASTAPI INIT
app = FastAPI(title="LLM Clause Retrieval", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"]
)

### REQUEST SCHEMA
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

### DOCUMENT PARSERS
def parse_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join([p.get_text() for p in doc])

def parse_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_email(raw: str) -> str:
    msg = message_from_string(raw)
    return "\n".join(part.get_payload(decode=True).decode(errors='ignore')
                     for part in msg.walk() if part.get_content_type() == "text/plain")

def fetch_and_parse_document(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch document")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    if url.endswith(".pdf"):
        return parse_pdf(tmp_path)
    elif url.endswith(".docx"):
        return parse_docx(tmp_path)
    elif url.endswith(".eml") or url.endswith(".msg"):
        return parse_email(response.text)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

### CHUNKING
def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    sents = text.split(". ")
    chunks, current = [], ""
    for sent in sents:
        if len(current) + len(sent) < chunk_size:
            current += sent + ". "
        else:
            chunks.append(current.strip())
            current = sent + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def insert_chunks(chunks: List[str]):
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[f"chunk-{i}"])

def query_chunks(question: str, k: int = 5) -> List[str]:
    result = collection.query(query_texts=[question], n_results=k)
    return result["documents"][0]

### PROMPTING & LLM
def build_prompt(question: str, context: List[str]) -> str:
    ctx = "\n---\n".join(context)
    return f"""
You are an insurance policy assistant. Use the context below to answer the user query.

Context:
{ctx}

Question:
{question}

Respond with this format:
{{"answer": "<your concise, explainable answer>"}}
"""

async def call_llm(prompt: str) -> str:
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2
    }
    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(LLM_API_URL, json=payload, headers=headers)
        res.raise_for_status()
        return res.json()["choices"][0]["text"].strip()

### API ROUTE
@app.post("/api/v1/hackrx/run")
async def run_query(req: QueryRequest):
    try:
        full_text = fetch_and_parse_document(req.documents)
        chunks = chunk_text(full_text)
        insert_chunks(chunks)

        final_answers = []
        for question in req.questions:
            top_chunks = query_chunks(question)
            prompt = build_prompt(question, top_chunks)
            response_text = await call_llm(prompt)
            try:
                response_json = json.loads(response_text)
                final_answers.append(response_json["answer"])
            except:
                final_answers.append("Could not extract answer. Raw: " + response_text)

        return {"answers": final_answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

### MAIN
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
