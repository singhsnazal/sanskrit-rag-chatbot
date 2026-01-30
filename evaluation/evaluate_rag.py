import os
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from rag_core.query_logic import ask_question

load_dotenv()

judge_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

eval_data = [
    {
        "question": "घण्टा वने कथम् अपतत् ?",
        "ground_truth": "चोरः व्याघ्रेण हतः तदा घण्टा वने अपतत्"
    },
    {
        "question": "चोरः कथं मृतः ?",
        "ground_truth": "चोरः व्याघ्रेण हतः"
    },
    {
        "question": "वानराः किं अकुर्वन् ?",
        "ground_truth": "वानराः घण्टां हस्ते धृत्वा अधुन्वन्"
    },
    {
        "question": "घण्टानादः कथम् अजायत् ?",
        "ground_truth": "वानरैः घण्टा अधुन्यमाना घण्टानादः अजायत्"
    },
    {
        "question": "जनाः किं अशङ्कन्त ?",
        "ground_truth": "जनाः अशङ्कन्त यत् शिखरप्रदेशे घण्टाकर्णः नाम राक्षसः वर्तते"
    },
    {
        "question": "राजा किम् अघोषयत् ?",
        "ground_truth": "यः घण्टाकर्णं नाशयेत् तस्मै राजा सुवर्णं दास्यति इति अघोषयत्"
    },
    {
        "question": "घण्टाकर्णः कुत्र वसति स्म ?",
        "ground_truth": "घण्टाकर्णः पर्वतस्य शिखरप्रदेशे वसति स्म"
    },
    {
        "question": "कः घण्टाकर्णस्य रहस्यं ज्ञातवान् ?",
        "ground_truth": "एका वृद्धा स्त्री घण्टाकर्णस्य रहस्यं ज्ञातवती"
    },
    {
        "question": "वृद्धा स्त्री किं दृष्टवती ?",
        "ground_truth": "सा दृष्टवती यत् वानराः घण्टां अधुन्वन्ति"
    },
    {
        "question": "वृद्धा स्त्री राजानं किम् अवदत् ?",
        "ground_truth": "वृद्धा स्त्री राजानं अवदत् यत् घण्टाकर्णः नास्ति केवलं वानराः घण्टां वादयन्ति"
    }
]


def judge_answer(question, answer, ground_truth):
    prompt = f"""
You are evaluating a Sanskrit Question Answering system.

QUESTION: {question}

GROUND TRUTH ANSWER:
{ground_truth}

MODEL ANSWER:
{answer}

Judge the model answer on:

1. Is the answer factually correct compared to ground truth?
2. Does the answer stay grounded in the story context?
3. Does it add hallucinated or unrelated information?

Respond ONLY in this format:

Correctness: 0 or 1  
Grounded: 0 or 1  
Hallucination: Yes or No
"""
    response = judge_llm.invoke(prompt).content.strip()
    return response


def parse_judgment(text):
    lines = text.splitlines()
    result = {"correctness": 0, "grounded": 0, "hallucination": "Yes"}

    for line in lines:
        if "Correctness" in line:
            result["correctness"] = int(line.split(":")[1].strip())
        elif "Grounded" in line:
            result["grounded"] = int(line.split(":")[1].strip())
        elif "Hallucination" in line:
            result["hallucination"] = line.split(":")[1].strip()

    return result


print("\n🧪 LLM JUDGE EVALUATION\n")

results = []

correct_total = 0
grounded_total = 0
hallucination_total = 0

for item in eval_data:
    result = ask_question(item["question"])
    answer = result["answer"]

    judgment_text = judge_answer(item["question"], answer, item["ground_truth"])
    judgment = parse_judgment(judgment_text)

    results.append({
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "model_answer": answer,
        "judgment": judgment
    })

    correct_total += judgment["correctness"]
    grounded_total += judgment["grounded"]
    hallucination_total += (1 if judgment["hallucination"] == "Yes" else 0)

    print(f"\nQ: {item['question']}")
    print(f"Model Answer: {answer}")
    print("Judge Result:", judgment)

# 📊 Overall Metrics
n = len(eval_data)
metrics = {
    "correctness_accuracy": round(correct_total / n, 2),
    "grounded_rate": round(grounded_total / n, 2),
    "hallucination_rate": round(hallucination_total / n, 2)
}

print("\n📊 FINAL METRICS")
print(metrics)

# 💾 Save JSON report
output = {
    "individual_results": results,
    "overall_metrics": metrics
}

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print("\n✅ Results saved to evaluation_results.json")
