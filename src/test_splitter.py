
from load_pdf import load_pdf
from split_docs import split_documents

docs = load_pdf("data/Amazon-2025-Annual-Report.pdf")

chunks = split_documents(docs)

print("Pages:", len(docs))
print("Chunks:", len(chunks))

print(chunks[0].page_content)
print(chunks[0].metadata)

for i in range(3):
    print("=" * 50)
    print(f"Chunk {i}")
    print(chunks[i].metadata)
    print(chunks[i].page_content[:300])