import streamlit as st
from src.load_pdf import load_pdf
from src.split_docs import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store

st.title("Finance AI assistant")

# st.write("RAG-based Finance Document Q&A System")

uploaded_file = st.file_uploader(
    "Upload Financial PDF",
    type = "pdf"
)

if uploaded_file:
    file_path = f"data/{uploaded_file.name}"

    with open(file_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    docs = load_pdf(file_path)

    chunks = split_documents(docs)

    st.success(f"PDF Loaded Successfully pages: {len(docs)}")
    # st.subheader("First Page Preview")
    st.write(f"Pages:{len(docs)}")
    st.write(f"Chunks created:{len(chunks)}")
    st.subheader("First Chunk")
    st.write(chunks[0].page_content[:1000])
    st.write(chunks[0].metadata)

    embeddings = get_embedding_model()

    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    st.success(f"FAISS Index Created with {len(chunks)} chunks")

    question = st.text_input("Test Retrieval Question")

    if question:

        docs = vector_store.similarity_search(
            question,
            k=3
        )

        st.subheader(
            "Retrieved Chunks"
        )

        for doc in docs:
            st.write(
                f"Page: {doc.metadata['page'] + 1}"
            )

            st.write(
                doc.page_content[:500]
            )

            st.divider()