*****************************************************************************************************************************************************************
How to Run the Sanskrit RAG Chatbot: 
1. Clone the Repository
   git clone https://github.com/YOUR_USERNAME/sanskrit-rag-chatbot.git
   cd sanskrit-rag-chatbot
2. Create Virtual Environment

   2.1 macOS / Linux
   python3 -m venv rag_env
   source rag_env/bin/activate

   2.2 Windows
   python -m venv rag_env
   rag_env\Scripts\activate

3. Install Dependencies
      pip install -r requirements.txt

4. Add Your Groq API Key

   Create a .env file in the project root:
   
   touch .env
   
   Add this inside:
   
   GROQ_API_KEY=your_groq_api_key_here
   get api key from : https://console.groq.com/home

5 .Add Sanskrit Documents
   
   Place your Sanskrit source document inside the data/ folder.
   
   Supported formats:
   
   .docx
   
   .txt
   
   .pdf
   
   Example:
   
   data/Rag-docs.docx

6. Run Document Ingestion (Build Search Index)

   This step chunks documents + creates FAISS + BM25 indexes
   
   python rag_core/ingest.py   

7. Start the FastAPI Server
   uvicorn api.main:app --reload --port 8000

******************************************************************************************************************************************************************


🕉️ Sanskrit RAG Chatbot  
### End-to-End Retrieval Augmented Generation System for Sanskrit Question Answering

This project implements an **End-to-End Retrieval Augmented Generation (RAG) chatbot** that answers questions in **Sanskrit** using a hybrid search system and Large Language Models.

It supports:
- Queries in **Sanskrit (Devanagari)**  
- Queries in **Transliteration (Latin script)**  
- Context-grounded responses using **Hybrid Retrieval (FAISS + BM25)**

## 🚀 System Overview ---<img width="1233" height="350" alt="indexing(part2)" src="https://github.com/user-attachments/assets/d6c3bc5f-279e-4291-89fa-c9471ee92192" />
<img width="981" height="1242" alt="rag_pipeline(2)" src="https://github.com/user-attachments/assets/84cf8223-1b1b-4bf3-9665-712a5c71d5ff" />




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
.
├── api
│   └── main.py
├── data
│   └── Rag-docs.docx
├── evaluation
│   └── evaluate_rag.py
├── evaluation_results.json
├── frontend
│   ├── static
│   │   └── style.css
│   └── templates
│       └── index.html
├── rag_core
│   ├── __init__.py
│   ├── bm25_docs.pkl
│   ├── check_index_dim.py
│   ├── faiss_index
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── ingest.py
│   ├── query_logic.py
│   ├── rag_pipeline.py
│   ├── test_retrieval_speed.py
│   └── transliterate.py
├── README.md
└── report

<img width="772" height="350" alt="Screenshot 2026-01-30 at 1 15 54 AM" src="https://github.com/user-attachments/assets/81d4ae73-8123-41dd-8074-19058711a84a" />


<img width="889" height="552" alt="Screenshot 2026-01-30 at 1 34 38 PM" src="https://github.com/user-attachments/assets/ef1d7f3d-a569-4697-84b0-1f75e9896cbd" />


<img width="889" height="552" alt="Screenshot 2026-01-30 at 1 29 50 PM" src="https://github.com/user-attachments/assets/9d255976-19c1-482e-ba6e-66a4fe39c7df" />








