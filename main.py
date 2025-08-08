# main.py
import os
import uvicorn
import fitz  # PyMuPDF
import docx
import requests
import tempfile
import faiss
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from email import message_from_string
from sentence_transformers import SentenceTransformer
import numpy as np
import httpx

### CONFIGURATION
LLM_API_URL = "http://localhost:1234/v1/completions"  # Example: LM Studio / Mistral
LLM_API_KEY = "none"  # No key needed for open source LM Studio API
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

### FASTAPI INIT
app = FastAPI(
    title="LLM-Powered Clause Retrieval System",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"]
)

### INPUT SCHEMA
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

### HELPER: Parse PDFs
def parse_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

### HELPER: Parse DOCX
def parse_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

### HELPER: Parse Emails
def parse_email(content: str) -> str:
    msg = message_from_string(content)
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload(decode=True).decode(errors='ignore'))
    return "\n".join(parts)

### DOWNLOAD AND PARSE DOCUMENT FROM URL
def fetch_and_parse_document(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch document")
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    if url.endswith(".pdf"):
        return parse_pdf(tmp_file_path)
    elif url.endswith(".docx"):
        return parse_docx(tmp_file_path)
    elif url.endswith(".eml") or url.endswith(".msg"):
        return parse_email(response.text)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

### SPLIT DOCUMENT TO CHUNKS
def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    sentences = text.split(". ")
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

### CREATE FAISS INDEX
def create_faiss_index(chunks: List[str]):
    embeddings = EMBED_MODEL.encode(chunks)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings, chunks

### QUERY FAISS INDEX
def query_faiss_index(question: str, index, chunks: List[str], embeddings, k=5):
    q_embedding = EMBED_MODEL.encode([question])[0]
    distances, indices = index.search(np.array([q_embedding]), k)
    return [chunks[i] for i in indices[0]]

### LLM INFERENCE
async def call_llm(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral",  # or any model hosted
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(LLM_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()

### BUILD PROMPT
def build_prompt(question: str, context: List[str]) -> str:
    context_str = "\n---\n".join(context)
    return f"""
You are an intelligent insurance policy assistant. Use the below context to answer the question.

Context:
{context_str}

Question:
{question}

Explain the reasoning and return a JSON response with the final answer as:
{{"answer": "<response here>"}}
"""

### API ROUTE
@app.post("/api/v1/hackrx/run")
async def run_query(req: QueryRequest):
    try:
        full_text = fetch_and_parse_document(req.documents)
        chunks = chunk_text(full_text)
        index, embeddings, chunk_texts = create_faiss_index(chunks)
        
        final_answers = []
        for question in req.questions:
            top_chunks = query_faiss_index(question, index, chunk_texts, embeddings)
            prompt = build_prompt(question, top_chunks)
            response_text = await call_llm(prompt)
            
            # Try parsing out answer from JSON
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
