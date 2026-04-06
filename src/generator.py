from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.retriever import retrieve

# --- Configuration ---
LLM_MODEL = "mistral"

def get_llm():
    """
    Connects to the locally running Ollama server
    and loads the Mistral 7B model.

    Why Ollama + Mistral?
    - Completely free, runs on your Mac
    - No API key, no internet needed after download
    - Mistral 7B is powerful enough for RAG tasks
    - Ollama serves it like a local API on port 11434
    """
    print(f"Connecting to Ollama — loading {LLM_MODEL}...")

    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0.1,   # Low temperature = more factual, less creative
                           # Good for RAG where we want accurate answers
        num_ctx=4096       # Context window size
    )

    print("   LLM ready!")
    return llm


def build_prompt_template():
    """
    The prompt template is the instruction we give Mistral.
    This is one of the most important parts of a RAG pipeline.

    Why does the prompt matter so much?
    Without clear instructions, the LLM might:
    - Make up information not in the documents
    - Ignore the context and answer from memory
    - Give overly long or unfocused answers

    Our prompt explicitly tells it to:
    - ONLY use the provided context
    - Admit when it doesn't know
    - Be concise and cite sources
    """
    template = """You are a helpful assistant that answers questions
based ONLY on the provided context from documents.

CONTEXT FROM DOCUMENTS:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer using ONLY the information in the context above
- If the context does not contain enough information, say so clearly
- Be concise and direct
- Mention which part of the document supports your answer

ANSWER:"""

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )


def format_context(docs: list) -> str:
    """
    Takes a list of Document objects and formats them
    into a single string to inject into the prompt.

    Why do we format them this way?
    The LLM needs to see the context as clean text.
    We add chunk numbers so the LLM can reference them.
    """
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(
            f"[Chunk {i+1} from {source}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def generate_answer(question: str) -> dict:
    """
    Full RAG pipeline in one function:
    1. Retrieve relevant chunks
    2. Format them as context
    3. Build the prompt
    4. Send to Mistral
    5. Return answer + sources

    Returns a dict with 'answer' and 'sources' keys
    so we can display them separately in the UI later.
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # Step 1: Retrieve relevant chunks
    print("\nStep 1: Retrieving relevant chunks...")
    docs = retrieve(question, k=3)

    if not docs:
        return {
            "answer": "I could not find relevant information in the documents.",
            "sources": []
        }

    # Step 2: Format context
    context = format_context(docs)

    # Step 3: Get LLM and prompt
    print("\n Step 2: Generating answer with Mistral...")
    llm = get_llm()
    prompt = build_prompt_template()

    # Step 4: Build and run the chain
    # This is LangChain's pipe operator — it chains steps together:
    # prompt → llm → output parser
    chain = prompt | llm | StrOutputParser()

    # Step 5: Generate answer
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    # Collect unique sources
    sources = list(set([
        doc.metadata.get("source", "unknown")
        for doc in docs
    ]))

    print(f"\nAnswer generated!")

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    # Make sure Ollama is running in another terminal first!
    # If not: open a new terminal and run: ollama serve

    test_questions = [
        "What is federated learning and why is it important in medical imaging?",
        "What ethical challenges does AI face in healthcare?"
    ]

    for question in test_questions:
        result = generate_answer(question)

        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(result["answer"])
        print(f"\nSources: {', '.join(result['sources'])}")
        print()