from .transliterate import to_devanagari, to_transliteration
from .rag_pipeline import load_rag_components
import time
import re


# Load once at startup (good for FastAPI performance)
retriever, llm = load_rag_components()


# 🔍 Detect if user typed in Latin script (transliteration)
def is_latin(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", text))


def ask_question(user_input: str):
    user_input = user_input.strip()

    if not user_input:
        return {
            "answer": "प्रश्नः न दत्तः",
            "retrieval_time": 0,
            "generation_time": 0
        }

    # 🔁 STEP 1 — Detect input script
    user_used_translit = is_latin(user_input)

    # Convert to Devanagari ONLY if transliteration
    query = to_devanagari(user_input) if user_used_translit else user_input
    formatted_query = f"query: {query}"

    # 🔍 STEP 2 — Retrieval
    start_retrieval = time.time()
    docs = retriever.invoke(formatted_query)
    end_retrieval = time.time()

    if not docs:
        answer = "न ज्ञायते"
        if user_used_translit:
            answer = to_transliteration(answer)

        return {
            "answer": answer,
            "retrieval_time": round(end_retrieval - start_retrieval, 3),
            "generation_time": 0
        }

    # 🔹 STEP 3 — Limit context size (reduce hallucination)
    context_chunks = [doc.page_content.strip()[:400] for doc in docs]
    context = "\n\n".join(context_chunks)

    # 🧠 STEP 4 — Strong grounding prompt
    prompt = f"""
You are an extractive Sanskrit Question Answering system.

STRICT RULES:
- Use ONLY the given CONTEXT
- Answer ONLY in Sanskrit
- Do NOT add outside knowledge
- Do NOT repeat sentences
- If answer is not clearly present, say exactly: न ज्ञायते

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER (2–3 short Sanskrit sentences):
"""

    # 🤖 STEP 5 — Generation
    start_gen = time.time()
    response = llm.invoke(prompt)
    answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
    end_gen = time.time()

    # 🔁 STEP 6 — Convert answer back if user used transliteration
    if user_used_translit:
        answer = to_transliteration(answer)

    return {
        "answer": answer,
        "retrieval_time": round(end_retrieval - start_retrieval, 3),
        "generation_time": round(end_gen - start_gen, 3)
     }
