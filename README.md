# 🕉️ Sanskrit RAG Chatbot  
### End-to-End Retrieval Augmented Generation System for Sanskrit Question Answering

This project implements an **End-to-End Retrieval Augmented Generation (RAG) chatbot** that answers questions in **Sanskrit** using a hybrid search system and Large Language Models.

It supports:
- Queries in **Sanskrit (Devanagari)**  
- Queries in **Transliteration (Latin script)**  
- Context-grounded responses using **Hybrid Retrieval (FAISS + BM25)**

---

## 🚀 System Overview

The system follows a complete RAG pipeline:

1. **User asks a question (Sanskrit or transliteration)**
2. FastAPI backend processes the request
3. Transliteration is converted to Devanagari (if needed)
4. Hybrid retrieval fetches relevant text chunks:
   - 🔎 Semantic Search → FAISS + Embeddings  
   - 🔤 Keyword Search → BM25
5. Context is passed to **Groq LLaMA 3.1 LLM**
6. LLM generates a **context-grounded Sanskrit answer**

---

## 🧠 Architecture

**Offline (Ingestion) Pipeline**
- Load Sanskrit documents (PDF / DOCX / TXT)
- Chunk using Recursive Character Splitting
- Generate embeddings using `intfloat/multilingual-e5-base`
- Store vectors in FAISS
- Store documents for BM25 keyword search

**Online (Query) Pipeline**
- User → FastAPI → Query Processing  
- Transliteration Handling  
- Hybrid Retrieval  
- Context Formation  
- LLM Answer Generation  
- Response returned to UI  

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| Backend API | FastAPI |
| Frontend | HTML + CSS |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Embeddings | intfloat/multilingual-e5-base |
| LLM | Groq LLaMA 3.1 |
| Evaluation | LLM-as-Judge |

---

## 📂 Project Structure

