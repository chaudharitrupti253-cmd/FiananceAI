from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)

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


PDF_PATH = "data/annual_report.pdf"

docs = load_pdf(PDF_PATH)
chunks = split_documents(docs)

embeddings = get_embedding_model()

vector_store = create_vector_store(
    chunks,
    embeddings
)

llm = load_llm()

df = pd.read_csv(
    "evaluation/test_questions.csv"
)

questions = []
answers = []
contexts = []
ground_truths = []

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

    questions.append(question)
    answers.append(answer)

    contexts.append(
        [
            doc.page_content
            for doc in source_docs
        ]
    )

    ground_truths.append(
        row["ground_truth"]
    )

dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
    ]
)

print(result)

result_df = result.to_pandas()

result_df.to_csv(
    "evaluation/ragas_results.csv",
    index=False
)

print(
    "Saved to evaluation/ragas_results.csv"
)