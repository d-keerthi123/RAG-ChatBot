# 🤖 RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built using LangChain and Ollama.  
The chatbot retrieves relevant information from documents and generates answers using a local LLM.

---

## 🚀 Features

- Chat with documents
- Local LLM using Ollama
- Vector search using FAISS
- Document embeddings using Sentence Transformers
- Simple UI using Streamlit

---

## 🛠️ Tech Stack

- Python
- LangChain
- Streamlit
- FAISS
- Ollama
- Sentence Transformers

---

## 📂 Project Structure

RAG-ChatBot/
│
├── app.py
├── data/
│   └── sample.txt
├── requirements.txt
└── README.md

---

## ▶️ Run the Project

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

---

## 🧠 How it Works

1. Document is loaded
2. Text is split into chunks
3. Embeddings are created
4. FAISS vector database stores embeddings
5. User asks a question
6. Relevant chunks are retrieved
7. LLM generates the answer
---
