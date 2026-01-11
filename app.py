from fastapi import FastAPI, HTTPException
import chromadb
import uuid
import os
from groq import Groq

app = FastAPI()


chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable not set")

groq_client = Groq(api_key=GROQ_API_KEY)


@app.get("/health")
@app.head("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "rag-api",
        "vector_store": "chromadb",
    }


@app.post("/query")
def query(q: str):
    try:
        results = collection.query(query_texts=[q], n_results=1)
        context = results["documents"][0][0] if results["documents"] else ""

        prompt = f"""
Context:
{context}

Question:
{q}

Answer clearly and concisely:
"""

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        answer = completion.choices[0].message.content
        return {"answer": answer}

    except Exception:
        raise HTTPException(
            status_code=503, detail="LLM service temporarily unavailable"
        )


@app.post("/add")
def add_knowledge(text: str):
    """Add new content to the knowledge base dynamically."""
    try:
        doc_id = str(uuid.uuid4())
        collection.add(documents=[text], ids=[doc_id])

        return {
            "status": "success",
            "message": "Content added to knowledge base",
            "id": doc_id,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
