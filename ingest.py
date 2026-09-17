"""Ingest the Apple 2025 Form 10-K PDF into Pinecone, one LangChain Document per PDF page."""

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PDF_PATH = os.path.join("data", "apple-2025-10k.pdf")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "apple-10k-rag")
CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
REGION = os.environ.get("PINECONE_REGION", "us-east-1")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def load_pages(pdf_path: str):
    """Load the PDF as one LangChain Document per page (no further splitting)."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    for doc in pages:
        # PyPDFLoader's "page" metadata is 0-indexed; add a human-readable 1-indexed page number.
        doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1
        doc.metadata["source"] = os.path.basename(pdf_path)
    return pages


def ensure_index(pc: Pinecone, name: str, dimension: int):
    if name not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
    while not pc.describe_index(name).status["ready"]:
        pass


def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"Expected PDF at {PDF_PATH}. Place the Apple 2025 Form 10-K PDF there first."
        )

    pages = load_pages(PDF_PATH)
    print(f"Loaded {len(pages)} pages from {PDF_PATH}")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    ensure_index(pc, INDEX_NAME, EMBEDDING_DIMENSION)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    PineconeVectorStore.from_documents(
        documents=pages,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    print(f"Upserted {len(pages)} page-documents into Pinecone index '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
