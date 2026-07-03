from load_pdf import load_pdf
from split_docs import split_documents
from embeddings import get_embedding_model
from vector_store import create_vector_store
from llm import load_llm
from rag_chain import (
    get_context,
    build_prompt,
    generate_answer
)

docs = load_pdf(
    "data\\Amazon-2025-Annual-Report.pdf"
)
print("Document load")
chunks = split_documents(docs)
print("Chunking")
embeddings = get_embedding_model()
print("Embedding")
vector_store = create_vector_store(
    chunks,
    embeddings
)
print("Vector store")
llm = load_llm()
print("Pass to LLM")
question = (
    "What was the company's revenue?"
)
print("Ask question")
context, sources = get_context(
    vector_store,
    question
)
print("Prompting")
prompt = build_prompt(
    context,
    question
)
print("Answering")
answer = generate_answer(
    llm,
    prompt
)

print(answer)
pages = set()

for doc in sources:
    pages.add(
        doc.metadata["page"] + 1
    )

print("Sources:", pages)