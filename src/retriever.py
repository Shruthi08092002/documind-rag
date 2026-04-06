from pathlib import Path
from src.embedder import load_vector_store, get_embedding_function

def get_retriever(k: int = 3):
    """
    Builds and returns a retriever object.

    What is a retriever?
    It's a wrapper around ChromaDB that knows how to:
    1. Take a plain text question
    2. Embed it using the same model we used for chunks
    3. Find the k most similar chunks
    4. Return them as LangChain Document objects

    Why k=3?
    3 chunks gives enough context without overwhelming the LLM.
    Too many chunks = LLM gets confused by too much information.
    Too few = LLM might miss key details.
    We'll experiment with this in Day 2.
    """
    vectorstore = load_vector_store()

    if vectorstore is None:
        print("Could not load vector store.")
        return None

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    print(f"Retriever ready — will return top {k} chunks per query")
    return retriever


def retrieve(question: str, k: int = 3) -> list:
    """
    Takes a question, finds the most relevant chunks.
    Returns a list of Document objects with text + metadata.

    This is the function our generator will call.
    """
    print(f"\nSearching for: '{question}'")

    retriever = get_retriever(k=k)
    if retriever is None:
        return []

    docs = retriever.invoke(question)

    print(f"\nFound {len(docs)} relevant chunks:\n")
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