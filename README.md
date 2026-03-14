# 🤖 RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built using LangChain and Ollama.  
The chatbot retrieves relevant information from documents and generates answers using a local LLM powered by Ollama .

---

## 🚀 Features
✔ Upload and chat with PDF documents  
✔ Retrieval-Augmented Generation (RAG) pipeline  
✔ Local LLM using **Ollama (TinyLlama)**  
✔ Vector similarity search using **FAISS**  
✔ Document embeddings using **Sentence Transformers**  
✔ Interactive UI built with **Streamlit**

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
├── requirements.txt
└── README.md

---

## ▶️ Run the Project

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

---

## 🧠 How It Works

1. User uploads a **PDF document**
2. Text is extracted using **pdfplumber**
3. The document is split into **smaller chunks**
4. Each chunk is converted into **vector embeddings**
5. Embeddings are stored in a **FAISS vector database**
6. When a user asks a question:
   - Relevant chunks are retrieved using **similarity search**
   - Context + question are passed to the **LLM**
7. The **LLM generates a final answer**
---
