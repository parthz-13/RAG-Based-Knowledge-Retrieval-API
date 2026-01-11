import chromadb

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("docs")

with open("knowledge.txt", "r") as f:
    text = f.read()

collection.add(documents=[text], ids=["knowledge"])

print("Embedding stored in Chroma")
