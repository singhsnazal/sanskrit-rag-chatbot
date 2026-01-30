import os
import pickle
from pathlib import Path

from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = BASE_DIR / "rag_core" / "faiss_index"
BM25_FILE = BASE_DIR / "rag_core" / "bm25_docs.pkl"

print("📚 Loading Sanskrit documents...")

# 🔍 Detect supported file
doc_path = None
for file in DATA_DIR.iterdir():
    if file.suffix.lower() in [".txt", ".docx", ".pdf"]:
        doc_path = file
        break

if not doc_path:
    raise FileNotFoundError("❌ No TXT, DOCX, or PDF file found in /data folder.")

print(f"Detected file: {doc_path.name}")

# 📄 Load based on type
if doc_path.suffix.lower() == ".pdf":
    loader = PyPDFLoader(str(doc_path))
elif doc_path.suffix.lower() == ".docx":
    loader = Docx2txtLoader(str(doc_path))
else:
    loader = TextLoader(str(doc_path), encoding="utf-8")

documents = loader.load()
print(f"📄 Loaded {len(documents)} document(s)")

# ✂️ Split into chunks
print("✂️ Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
docs = text_splitter.split_documents(documents)
print(f"✅ Total chunks created: {len(docs)}")

# 🔎 Embeddings
print("🔎 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# 🧠 Build FAISS
print("🧠 Building FAISS vector index...")
vectorstore = FAISS.from_documents(docs, embeddings)
FAISS_DIR.mkdir(parents=True, exist_ok=True)
vectorstore.save_local(str(FAISS_DIR))
print(f"✅ FAISS saved at: {FAISS_DIR}")

# 💾 Save docs for BM25 keyword retrieval
print("💾 Saving documents for BM25 retrieval...")
with open(BM25_FILE, "wb") as f:
    pickle.dump(docs, f)

print(f"✅ BM25 documents saved at: {BM25_FILE}")

print("🎉 Ingestion complete! Hybrid search ready.")
