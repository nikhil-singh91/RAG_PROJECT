"""
RAG Evaluation Framework (LLM-as-a-Judge and Retrieval Metrics).
Connects your PDF RAG pipeline with standard benchmark metrics:
- Hit Rate @ K
- Mean Reciprocal Rank (MRR)
- Groundedness / Faithfulness (LLM Judge)
- Correctness against Ground Truth (LLM Judge)
- Refusal Detection for out-of-domain queries
"""

import os
import json
import re
import time
from statistics import mean
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Import project RAG components
from config import config
from src.vector_db import VectorDatabase
from src.embeddings import EmbeddingManager
from src.retrieval import Retriever
from src.llm import GeminiRAGGenerator

load_dotenv()

# ----------------------------------------
# 1. GEMINI EVALUATOR SETUP
# ----------------------------------------

api_key = os.getenv("GEMINI_API_KEY", "").strip()
client = genai.Client(
    api_key=api_key,
    http_options={"api_version": "v1beta"},
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ----------------------------------------
# 2. RETRIEVAL METRICS
# ----------------------------------------

def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> int:
    """
    Checks whether at least one relevant document chunk
    appears in the top K retrieved documents.
    Returns 1 if hit, 0 otherwise.
    """
    if not relevant_ids:
        # If query has no relevant documents (e.g., negative test),
        # hit rate is 1 if nothing retrieved, 0 otherwise.
        return 0

    top_k = retrieved_ids[:k]
    return int(any(doc_id in relevant_ids for doc_id in top_k))


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Finds the reciprocal rank (1 / rank) of the first relevant document chunk.
    If the first relevant doc is at position 1 -> RR = 1.0.
    At position 2 -> RR = 0.5. If not found -> 0.0.
    """
    if not relevant_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


# ----------------------------------------
# 3. REFUSAL DETECTION
# ----------------------------------------

def detect_refusal(answer: str) -> bool:
    """
    Detects whether the chatbot appropriately refused to answer
    when requested information was absent from the context.
    """
    refusal_phrases = [
        "i cannot answer",
        "i can't answer",
        "i don't know",
        "could not find",
        "cannot find",
        "not available in the document",
        "not mentioned in the document",
        "i cannot help with that",
        "i'm unable to answer",
        "information is not provided",
        "context does not contain",
    ]

    answer_clean = answer.lower()
    return any(phrase in answer_clean for phrase in refusal_phrases)


# ----------------------------------------
# 4. GROUNDEDNESS AND CORRECTNESS (LLM JUDGE - BINARY 0 OR 1)
# ----------------------------------------

def llm_judge(question: str, context: str, answer: str, reference_answer: Optional[str] = None) -> Dict[str, Any]:
    """
    Uses Gemini to evaluate:
    1. Groundedness: Is the answer supported by context? (1 for Yes / Pass, 0 for No / Fail)
    2. Correctness: Is the answer factually correct? (1 for Yes / Pass, 0 for No / Fail)

    Scores are strictly binary: 0 or 1.
    """

    prompt = f"""You are a strict binary evaluator for a RAG chatbot.

Evaluate the following response using strictly binary scores (0 or 1).

QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context if context.strip() else "[No context retrieved]"}

REFERENCE ANSWER:
{reference_answer if reference_answer else "Not provided"}

CHATBOT ANSWER:
{answer}

Evaluate these two metrics strictly as binary integers (0 or 1):

1. Groundedness (0 or 1):
Are the claims in the chatbot answer strictly supported by the retrieved document context?
- Return 1 if ALL factual claims in the answer are supported by the context, or if the chatbot appropriately refused to answer because context was absent.
- Return 0 if the chatbot hallucinates, guesses, or states facts not found in the context.

2. Correctness (0 or 1):
Is the chatbot answer factually correct compared with the reference answer?
- Return 1 if the answer is factually accurate and aligns with the reference answer.
- Return 0 if the answer is incorrect or contradicts the reference answer.
- Return null ONLY if no reference answer is provided.

Return ONLY valid JSON in this exact format:

{{
    "groundedness": 0,
    "correctness": null,
    "reason": "Short explanation"
}}
"""

    # Retry loop with backoff for rate limit (429)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Deterministic judge
                )
            )

            result_text = response.text.strip()
            # Clean markdown code blocks
            result_text = re.sub(r"^```json\s*|^```\s*|\s*```$", "", result_text, flags=re.MULTILINE).strip()

            parsed = json.loads(result_text)

            # Ensure strict binary 0 or 1 output
            raw_groundedness = parsed.get("groundedness")
            groundedness_val = 1 if (raw_groundedness is not None and float(raw_groundedness) >= 0.5) else 0

            raw_correctness = parsed.get("correctness")
            if raw_correctness is not None:
                correctness_val = 1 if float(raw_correctness) >= 0.5 else 0
            else:
                correctness_val = None

            return {
                "groundedness": groundedness_val,
                "correctness": correctness_val,
                "reason": parsed.get("reason", "No reason provided."),
            }
        except Exception as e:
            err_str = str(e)
            if any(code in err_str for code in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                time.sleep(6)  # Wait for temporary demand spike or quota window to reset
                continue
            return {
                "groundedness": 0,
                "correctness": None,
                "reason": f"Judge error: {err_str}",
            }

    return {
        "groundedness": 0,
        "correctness": None,
        "reason": "Evaluation failed due to rate limits.",
    }


# ----------------------------------------
# 5. COMPLETE EVALUATION FOR ONE QUESTION
# ----------------------------------------

def evaluate_question(
    question: str,
    retrieved_docs: List[Dict[str, Any]],
    answer: str,
    relevant_doc_ids: List[str],
    reference_answer: Optional[str] = None,
    k: int = 5,
    cost: float = 0.0,
    delay: float = 0.0,
    was_refused: Optional[bool] = None,
) -> Dict[str, Any]:

    # Extract retrieved document IDs and combined text
    retrieved_ids = [doc["id"] for doc in retrieved_docs]
    context = "\n".join(doc["text"] for doc in retrieved_docs)

    # Retrieval metrics
    hit = hit_rate_at_k(retrieved_ids, relevant_doc_ids, k=k)
    mrr = reciprocal_rank(retrieved_ids, relevant_doc_ids)

    # Refusal metric
    refusal = detect_refusal(answer) if was_refused is None else was_refused

    # LLM evaluation (Groundedness & Correctness)
    judged = llm_judge(question, context, answer, reference_answer)

    return {
        "question": question,
        "hit_rate_at_k": hit,
        "mrr": round(mrr, 4),
        "groundedness": int(judged["groundedness"]),
        "correctness": int(judged["correctness"]) if judged["correctness"] is not None else None,
        "refusal": int(refusal),
        "cost": cost,
        "delay_seconds": round(delay, 3),
        "judge_reason": judged.get("reason", ""),
        "answer_preview": answer[:120] + "..." if len(answer) > 120 else answer,
    }


# ----------------------------------------
# 6. SUMMARY METRICS CALCULATION
# ----------------------------------------

def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:

    def average_metric(key: str) -> Optional[float]:
        values = [r[key] for r in results if r[key] is not None]
        return round(mean(values), 4) if values else None

    return {
        "Hit Rate at K": average_metric("hit_rate_at_k"),
        "MRR (Mean Reciprocal Rank)": average_metric("mrr"),
        "Groundedness": average_metric("groundedness"),
        "Correctness": average_metric("correctness"),
        "Refusal Rate": average_metric("refusal"),
        "Average Delay (seconds)": average_metric("delay_seconds"),
        "Total Questions Evaluated": len(results),
    }


# ----------------------------------------
# 7. DISPLAY RESULTS
# ----------------------------------------

def display_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:

    print("\n" + "=" * 65)
    print("📊 RAG EVALUATION REPORT (QUESTION-BY-QUESTION)")
    print("=" * 65)

    for i, res in enumerate(results, start=1):
        print(f"\n[Test Case {i}] Q: {res['question']}")
        print(f"  • Hit Rate@5  : {res['hit_rate_at_k']} (1 = Hit, 0 = Miss)")
        print(f"  • MRR         : {res['mrr']}")
        print(f"  • Groundedness: {res['groundedness']} (1 = Supported, 0 = Hallucination)")
        print(f"  • Correctness : {res['correctness']} (1 = Correct, 0 = Incorrect)" if res['correctness'] is not None else "  • Correctness : N/A")
        print(f"  • Refusal     : {res['refusal']} (1 = Refused, 0 = Answered)")
        print(f"  • Latency     : {res['delay_seconds']}s")
        print(f"  • Judge Notes : {res['judge_reason']}")

    summary = calculate_summary(results)

    print("\n" + "=" * 65)
    print("🏆 FINAL BENCHMARK SUMMARY")
    print("=" * 65)

    for metric, score in summary.items():
        print(f"  {metric:<30}: {score}")

    print("=" * 65 + "\n")
    return summary


# ----------------------------------------
# 8. INTEGRATED BENCHMARK RUNNER
# ----------------------------------------

def run_rag_evaluation(test_cases: Optional[List[Dict[str, Any]]] = None):
    """
    Executes the full evaluation pipeline using your live PDF RAG application.
    """
    print("Initializing RAG pipeline components for evaluation...")
    vector_db = VectorDatabase(
        persist_directory=config.chroma_persist_dir,
        collection_name=config.chroma_collection_name,
    )
    embedding_manager = EmbeddingManager.get_instance(config.default_embedding_model)
    retriever = Retriever(vector_db=vector_db, embedding_manager=embedding_manager)
    rag_generator = GeminiRAGGenerator(model_name="gemini-2.5-flash")

    # Default benchmark dataset based on 'Project Objectives.pdf'
    if test_cases is None:
        test_cases = [
            {
                "question": "What is the problem statement and key issues in the manual leave handling system?",
                "relevant_doc_ids": ["58917b01859cdd3f_Project Objectives_p2_c1"],
                "reference_answer": "Manual leave handling involves paperwork, scattered records, slow approval processes, and difficulty tracking employee leave balances.",
            },
            {
                "question": "What frontend technology is used in the technology stack and architecture?",
                "relevant_doc_ids": ["58917b01859cdd3f_Project Objectives_p4_c3"],
                "reference_answer": "The frontend is built using React to create pages that users see and interact with.",
            },
            {
                "question": "What results and deliverables does the Employee Leave Management System provide?",
                "relevant_doc_ids": ["58917b01859cdd3f_Project Objectives_p6_c5"],
                "reference_answer": "It provides centralized leave records and easier tracking for organizations.",
            },
            {
                "question": "What is the step-by-step recipe for baking a traditional chocolate cake?",
                "relevant_doc_ids": [],  # Intentionally out of domain
                "reference_answer": "I could not find information about baking chocolate cakes in the uploaded documents.",
            },
        ]

    evaluation_records = []

    print(f"\nRunning evaluation over {len(test_cases)} benchmark test cases...\n")

    for idx, test in enumerate(test_cases, 1):
        q = test["question"]
        expected_ids = test.get("relevant_doc_ids", [])
        ref_ans = test.get("reference_answer")

        print(f"[{idx}/{len(test_cases)}] Evaluating: '{q[:50]}...'")

        start_time = time.time()

        # Step 1: Retrieval
        retrieval_res = retriever.retrieve(query=q, top_k=5)
        retrieved_docs_formatted = [
            {"id": c.chunk_id, "text": c.text, "page": c.page_number}
            for c in retrieval_res.chunks
        ]

        # Step 2: Generation
        llm_res = rag_generator.generate_response(query=q, retrieved_chunks=retrieval_res.chunks)

        delay = time.time() - start_time

        # Step 3: Run Evaluation
        rec = evaluate_question(
            question=q,
            retrieved_docs=retrieved_docs_formatted,
            answer=llm_res.answer,
            relevant_doc_ids=expected_ids,
            reference_answer=ref_ans,
            k=5,
            delay=delay,
        )
        evaluation_records.append(rec)
        time.sleep(4)  # Small pause to respect Google AI Studio Free Tier rate limits (15 RPM)

    # Display results
    summary = display_results(evaluation_records)

    # Save to disk as JSON report
    output_file = "rag_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump({"summary": summary, "test_cases": evaluation_records}, f, indent=2)

    print(f"💾 Full evaluation report saved to: '{output_file}'")
    return evaluation_records, summary


if __name__ == "__main__":
    run_rag_evaluation()




    
