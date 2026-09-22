import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from ingest import CHROMA_DIR, EMBEDDING_MODEL

LLM_MODEL = "openai/gpt-oss-20b"

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using only the context below.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}
"""
)


def query_chunks(question: str, k: int = 4):
    if not Path(CHROMA_DIR).exists():
        raise FileNotFoundError(f"No database at {CHROMA_DIR}. Run ingest.py first.")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model,
        collection_name="rag_docs",
    )
    return vectorstore.similarity_search_with_score(question, k=k)


def answer_question(question: str, k: int = 4):
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    results = query_chunks(question, k=k)
    context = "\n\n".join(chunk.page_content for chunk, _score in results)

    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    response = (PROMPT | llm).invoke({"context": context, "question": question})
    return response.content, results


if __name__ == "__main__":
    load_dotenv()
    question = " ".join(sys.argv[1:]) or "What is Python?"
    answer, results = answer_question(question)

    print(f"Question: {question}\n")
    print("Answer:")
    print(answer)
    print()
    for i, (chunk, score) in enumerate(results, start=1):
        print(f"--- chunk {i} | score: {score:.4f} ---")
        print(chunk)
        print()
