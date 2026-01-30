import pickle
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def load_rag_components():
    BASE_DIR = Path(__file__).resolve().parent
    FAISS_PATH = BASE_DIR / "faiss_index"
    BM25_PATH = BASE_DIR / "bm25_docs.pkl"   # ✅ MATCHES YOUR FILE

    print(" Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )

    print(" Loading FAISS index...")
    vectorstore = FAISS.load_local(
        str(FAISS_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print(" Loading BM25 documents...")
    with open(BM25_PATH, "rb") as f:
        docs = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    print(" Creating hybrid retriever...")
    retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

    print(" Connecting to Groq LLM...")
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=250,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    return retriever, llm
