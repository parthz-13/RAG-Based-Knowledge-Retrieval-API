from fastapi import FastAPI, HTTPException, Body
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


@app.post(
    "/add",
    tags=["Knowledge"],
    summary="Add knowledge to the vector database",
    description=(
        "Adds new text content to the knowledge base. "
        "The text is embedded and stored in ChromaDB, making it "
        "available for semantic search during querying."
    ),
)
def add_knowledge(
    text: str = Body(
        ...,
        example="Google Antigravity is an AI-powered IDE developed by Google.",
    ),
):
    try:
        doc_id = str(uuid.uuid4())
        collection.add(documents=[text], ids=[doc_id])

        return {
            "status": "success",
            "message": "Content added to knowledge base",
            "id": doc_id,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add knowledge: {str(e)}",
        )


@app.post(
    "/query",
    tags=["Query"],
    summary="Ask a question over the knowledge base",
    description=(
        "Performs semantic search over stored knowledge using vector similarity "
        "and generates a context-aware answer using a large language model."
    ),
)
def query(
    q: str = Body(
        ...,
        example="What is Google Antigravity?",
    ),
):
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
            status_code=503,
            detail="LLM service temporarily unavailable",
        )
