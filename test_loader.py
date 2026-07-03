from src.load_pdf import load_pdf

docs = load_pdf("data\\Amazon-2025-Annual-Report.pdf")
print(f"Total pages :{len(docs)}")

print("\n First Page :\n")
print(docs[0].page_content[:1000])

print("\n Metadata")
print(docs[0].metadata)