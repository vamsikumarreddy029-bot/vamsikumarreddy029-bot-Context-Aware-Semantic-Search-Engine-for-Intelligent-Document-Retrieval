import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from utils.cleaning import clean_text
from utils.chunking import chunk_text
from utils.embedding import get_embeddings
from utils.vectorstore import create_vector_store, save_vector_store, load_vector_store


st.set_page_config(page_title="Semantic Search Engine", layout="wide")
st.title("📄 Context-Aware Semantic Search Engine")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Clean text
    for d in docs:
        d.page_content = clean_text(d.page_content)

    # Chunking
    chunks = chunk_text(docs)

    # Embeddings
    embeddings = get_embeddings()

    # Create FAISS
    vector_store = create_vector_store(chunks, embeddings)
    save_vector_store(vector_store)

    st.success("Document processed successfully!")

# Load existing DB
if os.path.exists("db"):
    embeddings = get_embeddings()
    vector_store = load_vector_store(embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use ONLY the context below to answer.
If not found, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    query = st.text_input("Ask a question:")

    if query:
        response = qa_chain.invoke(query)
        st.write("### Answer:")
        st.write(response["result"])