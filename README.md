# Pinecone RAG Homework — Apple 2025 Form 10-K

## Setup

1. Activate the venv and install dependencies:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in `PINECONE_API_KEY` and `OPENAI_API_KEY`.
3. Place the Apple 2025 Form 10-K PDF at `data/apple-2025-10k.pdf`.

## Run

Ingest the PDF (one Pinecone upsert per PDF page):
```bash
python ingest.py
```

Ask the two required questions and print answers with source page numbers:
```bash
python rag.py
```
