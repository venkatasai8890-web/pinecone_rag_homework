"""Week 6: generate RAG answers with the Week 5 Pinecone retriever and gpt-4o-mini, then judge them with LLM judges."""

import functools
import json
import os
import sys

# Make the repo root importable so we can reuse the Week 5 RAG code unchanged
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402

from rag import PROMPT, build_retriever_and_llm  # noqa: E402

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

QUESTIONS = [
    "What were Apple's total net sales for fiscal year 2025?",
    "How much net sales did iPhone generate in fiscal year 2025?",
    "What are Apple's reportable segments?",
    "How many full-time equivalent employees did Apple have at the end of fiscal year 2025?",
    "How did Apple's Greater China net sales change in fiscal year 2025 compared to 2024?",
]

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    "You are given a question, an answer and reference text. You must determine whether the given answer correctly answers the question based on the reference text. Here is the data:\n"
    "[BEGIN DATA]\n"
    "************\n"
    " [Question]: {question}\n"
    " ************\n"
    "[Reference]: {context}\n"
    " ************\n"
    " [Answer]: {sampled_answer}\n"
    "[END DATA]\n"
    "Your response must be a single word, either “correct” or “incorrect”, and should not contain any text or characters aside from that word. “correct” means that the question is correctly and fully answered by the answer. “incorrect” means that the question is not correctly or only partially answered by the answer"
)

VALID_VERDICTS = {"correct", "incorrect"}


@functools.cache
def get_retriever_and_llm():
    """Build the Pinecone retriever and gpt-4o-mini LLM once, on first use (not at import time)."""
    return build_retriever_and_llm()


@functools.cache
def get_judges():
    return {
        "judge_gemini_3_flash_preview": ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0),
        "judge_gpt_oss_120b": ChatGroq(model="openai/gpt-oss-120b", temperature=0),
    }


def answer_question(question: str) -> dict:
    """Same steps as rag.ask(), but returns the results instead of printing them."""
    retriever, llm = get_retriever_and_llm()
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = PROMPT.invoke({"context": context, "input": question})
    response = llm.invoke(messages)
    pages = sorted({int(doc.metadata["page_number"]) for doc in docs})
    return {
        "question": question,
        "context": context,
        "answer": response.content,
        "pages": pages,
    }


def normalize_verdict(raw: str) -> str:
    verdict = raw.strip().lower()
    if verdict.endswith("."):
        verdict = verdict[:-1].strip()
    return verdict if verdict in VALID_VERDICTS else "invalid"


def response_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text"
    )


def judge_answer(judge_llm, result: dict) -> str:
    messages = JUDGE_PROMPT.invoke(
        {
            "question": result["question"],
            "context": result["context"],
            "sampled_answer": result["answer"],
        }
    )
    response = judge_llm.invoke(messages)
    return normalize_verdict(response_text(response.content))


def save_results(results: list) -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_or_generate_results() -> list:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            results = json.load(f)
        if len(results) == len(QUESTIONS):
            print(f"Reusing {len(results)} saved results from {RESULTS_PATH}")
            return results

    results = []
    for i, question in enumerate(QUESTIONS, start=1):
        print(f"[{i}/{len(QUESTIONS)}] {question}")
        results.append(answer_question(question))
    save_results(results)
    print(f"Saved {len(results)} results to {RESULTS_PATH}")
    return results


def main():
    results = load_or_generate_results()
    judges = get_judges()
    for i, result in enumerate(results, start=1):
        for name, judge_llm in judges.items():
            key = f"{name}_verdict"
            if key in result:
                continue
            result[key] = judge_answer(judge_llm, result)
            print(f"[{i}/{len(results)}] {name}: {result[key]}")
        save_results(results)


if __name__ == "__main__":
    main()
