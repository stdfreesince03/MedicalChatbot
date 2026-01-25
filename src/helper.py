import os
import time
from typing import Optional

from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  

__all__ = ["init_rag"]

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384
DEFAULT_REGION = "us-east-1"
DEFAULT_K = 3


def init_rag(
    data_path: str,
    system_prompt: str,
    index_name: str = "medibot-rag",
    *,
    embedding_model: str = DEFAULT_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    upsert_batch_size: int = 25,
    top_k: int = DEFAULT_K,
) :

    load_dotenv()
    _require_env("PINECONE_API_KEY", "OPENAI_API_KEY")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = _ensure_index(pc, index_name, dimension=DEFAULT_DIM, region=DEFAULT_REGION)

    embedding = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )

    if _is_index_empty(index):
        print("Index is empty → ingesting PDF...")
        docs = _load_pdf_docs(data_path)
        chunks = _chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        _upsert_chunks(vectorstore, chunks, batch_size=upsert_batch_size)
        print("Ingestion complete")
    else:
        print("Index already has vectors → skipping ingestion")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=os.environ["OPENAI_API_KEY"], 
    )
    
    chat_history = ChatMessageHistory()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the user's past questions, rewrite it into a concise standalone search query. Do NOT answer."),
        MessagesPlaceholder(variable_name='chat_history'),
        ("human","{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm,retriever,history_aware_retriever_prompt)
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain,chat_history

def _require_env(*keys: str) -> None:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")


def _ensure_index(
    pc: Pinecone,
    index_name: str,
    *,
    dimension: int,
    region: str,
) :
    if not pc.has_index(index_name):
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region),
        )

    while not pc.describe_index(index_name).status["ready"]:
        print("Waiting for Pinecone index to be ready...")
        time.sleep(2)

    print("Index ready")
    return pc.Index(index_name)


def _is_index_empty(index) -> bool:
    stats = index.describe_index_stats()
    return stats.get("total_vector_count", 0) == 0


def _load_pdf_docs(data_path: str):
    loader = PyPDFLoader(data_path)
    docs = loader.load()

    docs = [d for d in docs if d.page_content and d.page_content.strip()]

    allowed = {"source", "page"}
    for d in docs:
        d.metadata = {k: v for k, v in d.metadata.items() if k in allowed}

    print(f"Loaded pages (non-empty): {len(docs)}")
    return docs


def _chunk_docs(docs, *, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunks created: {len(chunks)}")
    return chunks


def _upsert_chunks(vectorstore: PineconeVectorStore, chunks, *, batch_size: int = 25):
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        if (i // batch_size) % 10 == 0:
            print(f"Upserted {min(i+batch_size, total)}/{total} chunks...")
