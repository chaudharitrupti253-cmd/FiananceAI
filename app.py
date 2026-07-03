import streamlit as st
from src.load_pdf import load_pdf
from src.split_docs import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.llm import load_llm
from src.rag_chain import (get_context,build_prompt,generate_answer)
import os

# ----------------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Finance AI Assistant",
    page_icon="💰",
    layout="wide"
)

st.title("Finance AI assistant")
st.write("Upload a financial report and ask questions using RAG.")

# ----------------------------------------------------------------------- 
#  Cache Models 
#  ----------------------------------------------------------------------
@st.cache_resource
def get_cached_llm():
    return load_llm()

@st.cache_resource
def get_embeddings():
    return get_embedding_model()

# ----------------------------------------------------------------------- 
#  Initialize session State 
#  ----------------------------------------------------------------------
if "vector_store" not in st.session_state: 
    st.session_state.vector_store = None

if "pdf_name" not in st.session_state: 
    st.session_state.pdf_name = None

if "total_pages" not in st.session_state: 
    st.session_state.total_pages = 0 

if "total_chunks" not in st.session_state: 
    st.session_state.total_chunks = 0

# ----------------------------------- 
# Sidebar 
# ----------------------------------- 
with st.sidebar: 
    st.header("📊 Finance AI Assistant") 
    st.write("RAG-powered Financial Report Question Answering") 
    if st.session_state.pdf_name: 
        st.success( f"Loaded PDF:\n{st.session_state.pdf_name}" ) 
        st.metric( "Pages", st.session_state.total_pages ) 
        st.metric( "Chunks", st.session_state.total_chunks )

# ----------------------------------------------------------------------- 
#  PDF Upload 
#  ----------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Financial PDF",
    type = "pdf"
)

if uploaded_file is not None:
    # file_path = f"data/{uploaded_file.name}"
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join( "data", uploaded_file.name )

    # Save uploaded file
    with open(file_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    with st.spinner("Loading PDF..."):
        docs = load_pdf(file_path)

    st.success(f"PDF Loaded Successfully pages: {len(docs)}")
    st.write(f"Pages:{len(docs)}")

    # Split documents into chunks
    with st.spinner("Creating Documnet chunks..."):
        chunks = split_documents(docs)
    
    st.write(f"Total Chunks :{len(chunks)}")

    st.subheader("First Chunk")
    st.write(chunks[0].page_content[:1000])
    st.write(chunks[0].metadata)

    with st.spinner("Generating embeddings and building FAISS..."):
    
        embeddings = get_embedding_model()

        vector_store = create_vector_store(
            chunks,
            embeddings
        )
    # save session state
    st.session_state.vector_store = vector_store 
    st.session_state.pdf_name = uploaded_file.name
    st.session_state.total_pages = len(docs) 
    st.session_state.total_chunks = len(chunks)
    st.success(f"FAISS Index Created ")
    st.divider()

# ----------------------------------------------------------------------- 
#   Display current PDF 
#  ----------------------------------------------------------------------
    if st.session_state.pdf_name: 
        st.info( f"Current Document: "
                 f"{st.session_state.pdf_name}" )

# ----------------------------------------------------------------------- 
#   Ask Questions 
#  ----------------------------------------------------------------------

    question = st.text_input("Ask a question about the financial report")
    
    if question.strip():

        if st.session_state.vector_store is None:
            st.warning("Please upload a PDF First")
        else:
            vector_store = st.session_state.vector_store

            llm = get_cached_llm()

            try:
                with st.spinner("Generating answer..."):


                    # Retrieve context
                    context, source_docs = get_context(
                        vector_store,
                        question
                    )

                    # build prompt
                    prompt = build_prompt(
                        context,
                        question
                    )
                    
                    # Generate Answer
                    answer = generate_answer(
                        llm,
                        prompt
                    )
                    # ----------------------------------------------------------------------- 
                    #   Answer 
                    #  ----------------------------------------------------------------------

                    st.subheader("Answer")
                    st.info(answer)

                    # ----------------------------------------------------------------------- 
                    #   Source Pages 
                    #  ----------------------------------------------------------------------

                    st.subheader("Source Pages")

                    pages = sorted(
                        {
                            d.metadata["page"] + 1
                            for d in source_docs
                        }
                    )
                    for page in pages: 
                        st.write(f"📄 Page {page}")
                    

                    # ----------------------------------------------------------------------- 
                    #   Retrieve Chunks 
                    #  ----------------------------------------------------------------------

                    with st.expander("View Retrieved Context"):
                        for i, doc in enumerate(source_docs,start=1):
                            
                            st.markdown( f"### Chunk {i}" )
                            st.write(
                                f"Page {doc.metadata['page'] + 1}"
                            )

                            st.write(
                                doc.page_content[:1000]
                            )

                            st.divider()
            except Exception as e:
                st.error(f"Something went wrong:\n{str(e)}")
