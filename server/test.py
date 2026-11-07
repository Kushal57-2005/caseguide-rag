from app.services.retrieval_service import retrieve_documents
from app.services.generation_service import generate_answer


question = "How does the rainfall of Mahabaleshwar compare to Phaltan, and why is there a difference?"


print("\n1. Retrieving relevant documents...")

retrieved_chunks = retrieve_documents(
    question=question,
    top_k=5
)

print("Retrieved chunks:", len(retrieved_chunks))


print("\n2. Generating answer...")

result = generate_answer(
    question=question,
    retrieved_chunks=retrieved_chunks
)


print("\n========== RAG RESPONSE ==========")

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(result["answer"])

print("\nSources:")

for source in result["sources"]:
    print(
        f"File: {source['filename']} | "
        f"Page: {source['page_number']} | "
        f"Similarity: {source['similarity']:.4f}"
    )

print("\n==================================")