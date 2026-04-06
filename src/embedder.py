import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.ingestor import ingest

# --- Configuration ---
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embedding_function():
    """
    Creates and returns the embedding model.

    We use all-MiniLM-L6-v2 because:
    - It's free and runs completely locally (no API key)
    - It's small (80MB) but very capable
    - It's one of the most popular embedding models on HuggingFace
    - It converts text into 384-dimensional vectors

    Think of it as a translator that converts words into coordinates
    in a 384-dimensional space where similar meanings cluster together.
    """
    print(f" Loading embedding model: {EMBEDDING_MODEL}")
    print("    (First run downloads ~80MB — subsequent runs are instant)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Use CPU — no GPU needed
        encode_kwargs={"normalize_embeddings": True}
        # normalize=True means all vectors have length 1
        # This makes similarity comparisons more accurate
    )
    return embeddings


def build_vector_store(chunks: list, persist_path: str = CHROMA_DB_PATH):
    """
    Takes the document chunks and:
    1. Embeds each chunk into a vector using the embedding model
    2. Stores all vectors in ChromaDB on disk

    ChromaDB is our vector database — think of it like a regular
    database but instead of searching by exact keywords, it searches
    by MEANING (semantic similarity).
    """
    if not chunks:
        print("No chunks provided. Run ingestor first.")
        return None

    print(f"\n Embedding {len(chunks)} chunks into vectors...")
    print(f"    This converts each chunk of text into 384 numbers")
    print(f"    that capture its semantic meaning.")

    embedding_fn = get_embedding_function()

    # Create the ChromaDB vector store
    # This does three things at once:
    # 1. Embeds every chunk using our model
    # 2. Stores the vectors in ChromaDB
    # 3. Saves everything to disk at CHROMA_DB_PATH
    print(f"\n💾 Storing vectors in ChromaDB at: {persist_path}/")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=persist_path
    )

    # Count how many vectors were stored
    count = vector_store._collection.count()
    print(f"Successfully stored {count} vectors in ChromaDB!")
    print(f"Database saved to: {persist_path}/")
    print(f"Each vector has 384 dimensions")

    return vector_store


def load_vector_store(persist_path: str = CHROMA_DB_PATH):
    """
    Loads an existing ChromaDB database from disk.
    This is used when you want to query without re-embedding.

    Why is this useful? Embedding 241 chunks takes ~10-30 seconds.
    Once done, we save to disk and can reload instantly next time.
    """
    if not os.path.exists(persist_path):
        print(f" No database found at {persist_path}")
        print("    Run build_vector_store() first to create it")
        return None

    print(f" Loading existing ChromaDB from: {persist_path}/")
    embedding_fn = get_embedding_function()

    vector_store = Chroma(
        persist_directory=persist_path,
        embedding_function=embedding_fn
    )

    count = vector_store._collection.count()
    print(f"Loaded vector store with {count} vectors")
    return vector_store


def embed(data_path: str = "data/raw", persist_path: str = CHROMA_DB_PATH):
    """
    Main function — runs the full embedding pipeline:
    Ingest → Chunk → Embed → Store
    """
    print("=" * 50)
    print("STEP 1: Loading and chunking documents")
    print("=" * 50)
    chunks = ingest(Path(data_path))

    if not chunks:
        print("No chunks to embed. Add documents to data/raw/")
        return None

    print("\n" + "=" * 50)
    print("STEP 2: Embedding chunks into ChromaDB")
    print("=" * 50)
    vector_store = build_vector_store(chunks, persist_path)

    return vector_store


if __name__ == "__main__":
    vector_store = embed()

    if vector_store:
        print("\n" + "=" * 50)
        print("Embedding complete!")
        print("=" * 50)
        print(f"Your documents are now searchable by meaning.")
        print(f"ChromaDB is saved at: {CHROMA_DB_PATH}/")
        print(f"\nNext step: Run src/retriever.py to test search!")