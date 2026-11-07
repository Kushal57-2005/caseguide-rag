import chromadb

client = chromadb.PersistentClient(
    path="./storage/chroma_db"
)

try:
    client.delete_collection("pdf_documents")
    print("Old collection deleted!")
except Exception as e:
    print("Error:", e)

collection = client.get_or_create_collection(
    name="pdf_documents",
    metadata={
        "hnsw:space": "cosine"
    }
)

print("New collection created!")
print("Collection count:", collection.count())