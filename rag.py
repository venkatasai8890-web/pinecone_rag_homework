"""Answer the two required questions using LangChain retrieval over the Pinecone index."""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "apple-10k-rag")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

QUESTIONS = [
    "Can you tell me Apple revenue by products and services?",
    "Where is Apple headquarters?",
]

PROMPT = ChatPromptTemplate.from_template(
    "You are answering questions about Apple's 2025 Form 10-K filing. "
    "Use only the following excerpts to answer. If the answer isn't in the excerpts, say so.\n\n"
    "Excerpts:\n{context}\n\nQuestion: {input}"
)


def build_retriever_and_llm():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return retriever, llm


def ask(retriever, llm, question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = PROMPT.invoke({"context": context, "input": question})
    response = llm.invoke(messages)
    pages = sorted({int(doc.metadata["page_number"]) for doc in docs})
    print(f"Q: {question}")
    print(f"A: {response.content}")
    print(f"Source pages: {pages}")
    print()


def main():
    retriever, llm = build_retriever_and_llm()
    for question in QUESTIONS:
        ask(retriever, llm, question)


if __name__ == "__main__":
    main()
