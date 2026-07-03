def get_context(vector_store, question):

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context, docs

def build_prompt(
        context,
        question
):

    prompt = f"""
You are a financial document assistant.

Answer ONLY from the context.

If information is not available,
say:
'I could not find this information in the documents.'

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def generate_answer(llm,prompt):

    response = llm.invoke(prompt)

    answer = response.content

    return answer