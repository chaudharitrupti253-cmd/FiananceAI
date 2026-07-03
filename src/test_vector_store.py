from load_pdf import load_pdf
from split_docs import split_documents
from embeddings import get_embedding_model
from vector_store import create_vector_store

docs = load_pdf("data\\Amazon-2025-Annual-Report.pdf")
chunks = split_documents(docs)

embeddings = get_embedding_model()

vector_store = create_vector_store(
    chunks,
    embeddings
)

print("Documents: ",len(docs))
print("Chunks",len(chunks))
print("FAISS created successfully")

query = "What was the company's revenue?"

results = vector_store.similarity_search(
    query,
    k=3
)

for doc in results:
    print("=" * 50)
    print(doc.metadata)
    print(doc.page_content[:300])
