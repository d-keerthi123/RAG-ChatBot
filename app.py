import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

load_dotenv()

st.title("RAG Chatbot 🤖")

# Load document
loader = TextLoader("data/sample.txt")
documents = loader.load()

# Split document
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
vectorstore = FAISS.from_documents(docs, embeddings)

# Retriever
retriever = vectorstore.as_retriever()

# LLM
llm = Ollama(model="tinyllama")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# User input
query = st.text_input("Ask something")

if query:
    result = qa_chain.run(query)
    st.write(result)