import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from src.embedder import load_vector_store, get_embedding_function


def get_retriever(k: int = 3, search_type: str = "similarity"):
    """
    Builds and returns a retriever.

    search_type options:
    - "similarity" — standard cosine similarity search
    - "mmr" — Maximum Marginal Relevance
      finds relevant AND diverse chunks
      avoids returning 3 chunks that all say the same thing
    """
    vectorstore = load_vector_store()

    if vectorstore is None:
        print("❌ Could not load vector store.")
        return None

    if search_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3}
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    print(f"✅ Retriever ready — {search_type}, top {k} chunks")
    return retriever


def retrieve(question: str, k: int = 3, search_type: str = "similarity") -> list:
    """
    Takes a question, finds the most relevant chunks.
    Returns a list of Document objects.
    """
    print(f"\n🔍 Searching for: '{question}'")

    retriever = get_retriever(k=k, search_type=search_type)
    if retriever is None:
        return []

    docs = retriever.invoke(question)

    print(f"\n📄 Found {len(docs)} relevant chunks:\n")
    for i, doc in enumerate(docs):
        print(f"  Chunk {i+1}:")
        print(f"  Source: {doc.metadata.get('source', 'unknown')}")
        print(f"  Preview: {doc.page_content[:200]}...")
        print()

    return docs


if __name__ == "__main__":
    test_questions = [
        "What is federated learning?",
        "How do transformers work in medical imaging?",
        "What are the ethical challenges of AI in healthcare?"
    ]

    for question in test_questions:
        print("=" * 60)
        retrieve(question)