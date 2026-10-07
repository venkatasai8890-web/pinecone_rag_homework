# Week 6: Evaluate RAG Responses with LLM-as-a-Judge

## Objective

Evaluate the answers produced by the Week 5 RAG pipeline using two LLM judges, each returning a single verdict per answer.

## Setup

- **Source document:** Apple 2025 Form 10-K (`data/apple-2025-10k.pdf`)
- **Vector database:** Pinecone (index `apple-10k-rag`)
- **Embedding model:** `text-embedding-3-small`
- **Initial RAG response model:** `gpt-4o-mini`
- **Judge 1:** `gemini-3-flash-preview` (Google Gemini)
- **Judge 2:** `openai/gpt-oss-120b` (Groq)

## Evaluation

Five questions were run through the Week 5 RAG pipeline. For each one, the saved question, retrieved context, and `gpt-4o-mini` answer were sent to both judges.

Each judge receives:
- the question,
- the retrieved reference context,
- the initial RAG answer.

Each judge must respond with only `correct` or `incorrect`. Responses are normalized (lowercased, whitespace and a trailing period removed); anything else is recorded as `invalid`.

Prompt used for both judges (from the homework specification):

> You are given a question, an answer and reference text. You must determine whether the given answer correctly answers the question based on the reference text. ... Your response must be a single word, either "correct" or "incorrect", and should not contain any text or characters aside from that word.

## Result

Both judges marked all 5 answers `correct`: 10/10 valid judge verdicts, with no `invalid` results.

Per-question verdicts, the original RAG answers, and retrieved contexts are stored in [results.json](results.json).

## Setup and Run

1. From the repository root, activate the virtual environment and install dependencies:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in `PINECONE_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, and `GOOGLE_API_KEY`. Never commit `.env`.
3. Make sure the Pinecone index has been built with `python ingest.py` (see the root README).
4. Run the evaluation from the repository root:
   ```bash
   python week6_llm_judge/evaluate_rag.py
   ```

If `results.json` already contains all 5 answers, the script reuses them and only runs the judges. Judge verdicts already stored in a result are skipped, so delete the `judge_*` fields to re-judge.
