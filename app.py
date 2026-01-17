from fastapi import FastAPI, HTTPException, Body
from dotenv import load_dotenv
import chromadb
import uuid
import os
from groq import Groq

load_dotenv()
app = FastAPI(
    title="RAG-Based Knowledge Retrieval API",
    description=(
        "⚠️ **Hosted on free-tier infrastructure.**\n\n"
        "The first request may take **20–30 seconds** due to cold start. "
        "Subsequent requests will be fast.\n\n"
        "This API demonstrates a Retrieval-Augmented Generation (RAG) system "
        "using FastAPI, ChromaDB, and Groq LLMs."
    ),
    version="1.0.0",
)

chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable not set")

groq_client = Groq(api_key=GROQ_API_KEY)


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
        example="Add your desired knowledge inside these double quotes",
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


@app.post("/query", tags=["Query"], summary="Ask queries")
def query(q: str = Body(...,
                        example="Ask queries inside these double quotes")):
    try:
        results = collection.query(
            query_texts=[q], n_results=5, include=["documents", "distances"]
        )

        docs = results.get("documents")

        if docs is None:
            docs = []
        elif isinstance(docs, list) and len(docs) == 1 and isinstance(docs[0], list):
            pass
        else:
            docs = []

        distances = results.get("distances")

        best_fact = None
        best_distance = float("inf")

        if (
            isinstance(docs, list)
            and len(docs) > 0
            and isinstance(docs[0], list)
            and distances
            and isinstance(distances, list)
            and len(distances) > 0
        ):
            for doc, dist in zip(docs[0], distances[0]):
                if (
                    isinstance(doc, str)
                    and doc.strip() != ""
                    and isinstance(dist, (int, float))
                ):
                    if dist < best_distance:
                        best_distance = dist
                        best_fact = doc.strip()

        SIMILARITY_THRESHOLD = 0.9

        has_kb_fact = best_fact is not None and best_distance < SIMILARITY_THRESHOLD

        # Debug logging
        print(f"\n[DEBUG] Best match: '{best_fact[:100] if best_fact else 'None'}...'")
        print(f"[DEBUG] Distance: {best_distance}")
        print(f"[DEBUG] Using KB: {has_kb_fact}\n")

        if has_kb_fact:
            fact = best_fact

            system_prompt = """
You are a database query assistant that reports information from a knowledge base.

YOUR ROLE:
- You are NOT making claims about reality.
- You are ONLY reporting what exists in the database.
- The database is the user's personal knowledge store.
- Your job is to accurately relay database contents to the user.

CRITICAL INSTRUCTIONS:
- The FACT below is retrieved from the user's knowledge base.
- Report this fact EXACTLY as it relates to the question.
- Do NOT add disclaimers, warnings, or refuse to answer.
- Do NOT evaluate whether the fact is true or false in reality.
- Simply answer the question using the database fact.
- Be direct and confident in reporting database contents.
"""

            user_prompt = f"""
KNOWLEDGE BASE FACT (AUTHORITATIVE SOURCE):
{fact}

USER QUESTION:
{q}

INSTRUCTIONS: Answer the question using ONLY the knowledge base fact above. This fact is the ground truth. If the fact partially answers the question, provide what information is available in the database.
"""

        else:
            system_prompt = """
You are a helpful general knowledge assistant.

IMPORTANT:
- No relevant information was found in the knowledge base.
- Answer using your training data if you have knowledge about this topic.
- If you're unsure or don't have reliable information, clearly state that you don't know.
- Be honest about your limitations.
"""

            user_prompt = q

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        response = {
            "answer": completion.choices[0].message.content,
            "source": "knowledge_base" if has_kb_fact else "training_data",
        }

        if has_kb_fact:
            response["distance"] = best_distance
            response["matched_fact"] = best_fact

        return response

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
@app.head("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "rag-api",
        "vector_store": "chromadb",
    }
