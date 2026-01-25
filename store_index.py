import time
start = time.time()

print("A) loading env...")
from dotenv import load_dotenv
load_dotenv()
print("   done")

print("B) imports...")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # FIXED
from pinecone import Pinecone, ServerlessSpec
import os
print("   done")

# -----------------------------
print("C) loading PDF...")
file_path = './data/medical-book.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()
print(f"   pages loaded: {len(docs)}")

# -----------------------------
print("D) filtering empty pages...")
filtered_docs = [
    d for d in docs
    if d.page_content and d.page_content.strip()
]
print(f"   pages after filter: {len(filtered_docs)}")

ALLOWED_META = ['source', 'page']
for d in filtered_docs:
    d.metadata = {k: v for k, v in d.metadata.items() if k in ALLOWED_META}

docs = filtered_docs

# -----------------------------
print("E) chunking...")
chunker = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = chunker.split_documents(docs)
print(f"   chunks created: {len(chunks)}")

# -----------------------------
print("F) loading embedding model (HF)...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=model_name)
print("   embedding model ready")

# -----------------------------
print("G) setting API keys...")
print("   PINECONE:", bool(os.getenv("PINECONE_API_KEY")))
print("   OPENAI:", bool(os.getenv("OPENAI_API_KEY")))

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# -----------------------------
print("H) checking / creating Pinecone index...")
index_name = "medibot-rag"

if not pc.has_index(index_name):
    print("   index not found, creating...")
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("   index created")
else:
    print("   index already exists")

print(f"\n✅ DONE in {time.time() - start:.2f} seconds")
