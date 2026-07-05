import os
import sys
import glob
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


CHROMA_DIR = Path("chroma_db")
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(data_dir: str) -> List[Document]:
    docs = []
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: directory '{data_dir}' not found")
        sys.exit(1)

    pdf_files = list(data_path.glob("**/*.pdf"))
    txt_files = list(data_path.glob("**/*.txt"))

    for fp in pdf_files:
        loader = PyPDFLoader(str(fp))
        docs.extend(loader.load())

    for fp in txt_files:
        loader = TextLoader(str(fp), encoding="utf-8")
        docs.extend(loader.load())

    if not docs:
        print(f"No PDF or TXT files found in '{data_dir}'")
        sys.exit(1)

    print(f"Loaded {len(docs)} document(s)")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


class RAGSystem:
    def __init__(self, force_reindex: bool = False, debug: bool = False):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
        )
        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )
        self.vector_store = None
        self.retriever = None
        self.force_reindex = force_reindex
        self.debug = debug

    def index_documents(self, data_dir: str) -> None:
        if CHROMA_DIR.exists() and not self.force_reindex:
            print("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=self.embeddings,
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            print(f"Vector store loaded ({CHROMA_DIR})")
            return

        docs = load_documents(data_dir)
        chunks = chunk_documents(docs)

        print("Creating embeddings and building vector store...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        print(f"Vector store created with {len(chunks)} chunks")

    def query(self, question: str) -> str:
        if self.retriever is None:
            return "Error: No documents indexed. Call index_documents() first."

        retrieved_docs = self.retriever.invoke(question)

        if self.debug:
            print("\n--- Retrieved Contexts ---")
            for i, d in enumerate(retrieved_docs):
                src = d.metadata.get("source", "unknown")
                print(f"[{i+1}] (from {src}):\n{d.page_content[:200]}...\n")
            print("--- End Contexts ---\n")

        context = "\n\n".join(d.page_content for d in retrieved_docs)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. Answer the question using ONLY the context below. "
                "If the context does not contain the answer, say \"I cannot find this in the provided documents.\" "
                "Do not use your own knowledge. Be concise.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            ),
        ])

        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)

    def interactive(self) -> None:
        print("\n" + "=" * 60)
        print("  RAG Question Answering System")
        print("  Type 'quit' or 'exit' to stop")
        print("=" * 60)
        while True:
            try:
                q = input("\nQuestion: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if not q:
                    continue
                print("\nThinking...")
                answer = self.query(q)
                print(f"\nAnswer: {answer}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Document Question Answering System"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data",
        help="Directory containing PDF/TXT documents (default: 'data')",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindexing even if vector store exists",
    )
    parser.add_argument(
        "--query", "-q",
        help="Single query to run (non-interactive mode)",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Show retrieved contexts",
    )

    args = parser.parse_args()

    rag = RAGSystem(force_reindex=args.reindex, debug=args.debug)
    rag.index_documents(args.data_dir)

    if args.query:
        answer = rag.query(args.query)
        print(answer)
    else:
        rag.interactive()


if __name__ == "__main__":
    main()
