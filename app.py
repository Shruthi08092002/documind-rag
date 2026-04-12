import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from src.generator import generate_answer

# --- Page config ---
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="centered"
)

# --- Header ---
st.title("🧠 DocuMind")
st.caption("Ask questions about your documents — powered by local RAG")
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.markdown("""
    **DocuMind** is a local RAG pipeline that answers
    questions grounded in your documents.

    **Stack:**
    - LangChain + ChromaDB
    - sentence-transformers embeddings
    - Mistral 7B via Ollama
    - RAGAS evaluation

    **How it works:**
    1. Your question is embedded into a vector
    2. ChromaDB finds the 3 most similar chunks
    3. Mistral generates an answer from those chunks
    """)

    st.divider()
    st.header("Loaded documents")
    try:
        from src.embedder import load_vector_store
        vs = load_vector_store()
        count = vs._collection.count()
        st.success(f"{count} chunks indexed")
    except Exception:
        st.error("Could not load vector store")

    st.divider()
    st.header("Try asking")
    st.markdown("""
    - What is federated learning?
    - What is Grad-CAM?
    - What ethical challenges does AI face?
    - What is the EchoNet dataset?
    - What are the main tasks in medical image analysis?
    """)

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.caption(f"📄 {source}")

# --- Chat input ---
if prompt := st.chat_input("Ask a question about your documents..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = generate_answer(prompt)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                with st.expander("Sources"):
                    for source in sources:
                        st.caption(f"📄 {source}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f"Error generating answer: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })