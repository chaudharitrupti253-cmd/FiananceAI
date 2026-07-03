import pandas as pd

from src.load_pdf import load_pdf
from src.split_docs import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.llm import load_llm
from src.rag_chain import (
    get_context,
    build_prompt,
    generate_answer
)


PDF_PATH = "data\\Amazon-2025-Annual-Report.pdf"


print("Loading PDF...")
docs = load_pdf(PDF_PATH)

print("Creating chunks...")
chunks = split_documents(docs)

print("Loading embedding model...")
embeddings = get_embedding_model()

print("Creating vector store...")
vector_store = create_vector_store(
    chunks,
    embeddings
)

print("Loading LLM...")
llm = load_llm()

df = pd.read_csv(
    "evaluation\\test_questions.csv"
)

predictions = []

for _, row in df.iterrows():

    question = row["question"]

    context, source_docs = get_context(
        vector_store,
        question
    )

    prompt = build_prompt(
        context,
        question
    )

    answer = generate_answer(
        llm,
        prompt
    )

    predictions.append(answer)

df["predicted_answer"] = predictions

# Very simple exact match score
df["correct"] = (
    df["ground_truth"]
    .str.lower()
    .str.strip()
    ==
    df["predicted_answer"]
    .str.lower()
    .str.strip()
)

accuracy = (
    df["correct"].mean()
) * 100

print("\nEvaluation Complete")
print(f"Accuracy: {accuracy:.2f}%")

df.to_csv(
    "evaluation/results.csv",
    index=False
)

print(
    "Saved results to evaluation/results.csv"
)