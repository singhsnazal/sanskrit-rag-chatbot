import time
import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🔹 Load E5 embedding model
print("🔎 Loading embedding model (E5)...")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# 🔹 Confirm embedding dimension
test_vec = embeddings.embed_query("test")
print("Embedding dimension:", len(test_vec))

# 🔹 Load FAISS index
print("📂 Loading FAISS index...")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

print(f"📊 Total vectors in index: {db.index.ntotal}")

# 🔹 Sanskrit query (E5 models prefer 'query:' prefix)
query = "query: मूर्खभृत्यस्य कथा का"
print(f"\n🧠 Running retrieval for query: {query}")

# ⏱ Retrieval timing
start_time = time.time()
docs_and_scores = db.similarity_search_with_score(query, k=2)
end_time = time.time()

print(f"\n⏱ Retrieval Time: {end_time - start_time:.4f} seconds")

# 🔹 Display retrieved chunks (cleaned)
for i, (doc, score) in enumerate(docs_and_scores):
    print(f"\n📜 Result {i+1}  |  Similarity Score: {score:.4f}")
    
    # Clean extra blank lines for nicer display
    cleaned_text = re.sub(r'\n\s*\n+', '\n\n', doc.page_content).strip()
    
    print(cleaned_text[:600])
