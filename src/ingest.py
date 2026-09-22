import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents(docs_dir: Path | str = DOCS_DIR):
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

    loaders = [
        DirectoryLoader(
            str(docs_dir), # look inside this folder
            glob="**/*.txt", # keep files that match this pattern
            loader_cls=TextLoader, # for each match, use TextLoader
            loader_kwargs={"encoding": "utf-8"},
        ),
        DirectoryLoader(
            str(docs_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        ),
        DirectoryLoader(
            str(docs_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        ),
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load()) # .load() command will load all the matching documents for a particular loader and will put them in a list. Extend will add the documents to the main documents list while unpacking the documents into the list.
    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def store_chunks(chunks, persist_directory: Path | str = CHROMA_DIR):
    persist_directory = Path(persist_directory)
    if persist_directory.exists():
        shutil.rmtree(persist_directory)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(persist_directory),
        collection_name="rag_docs",
    )
    return vectorstore


if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s) from {DOCS_DIR}\n")

    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)\n")

    print("Creating embeddings and saving to Chroma...")
    store_chunks(chunks)
    print(f"Stored {len(chunks)} chunk(s) in {CHROMA_DIR}")
