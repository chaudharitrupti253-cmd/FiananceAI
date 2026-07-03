from langchain_community.vectorstores import FAISS

def create_vector_store(chunks,embeddings):

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_store.save_local("faiss_index")
    
    return vector_store