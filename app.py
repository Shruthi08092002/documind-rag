import warnings
from pathlib import Path 
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
    st.header("Upload documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Index uploaded documents", type="primary"):
            with st.spinner("Indexing documents..."):
                try:
                    raw_path = Path("data/raw")
                    raw_path.mkdir(parents=True, exist_ok=True)

                    for file in uploaded_files:
                        save_path = raw_path / file.name
                        with open(save_path, "wb") as f:
                            f.write(file.getbuffer())

                    from src.ingestor import ingest
                    from src.embedder import build_vector_store, get_embedding_function

                    chunks = ingest(raw_path)
                    embedding_fn = get_embedding_function()
                    build_vector_store(chunks, embedding_fn)

                    st.success(f"Indexed {len(uploaded_files)} document(s) — {len(chunks)} chunks!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error indexing: {e}")

    st.divider()
    st.header("About")
    st.markdown("""
    **DocuMind** is a local RAG pipeline that answers
    questions grounded in your documents.

    **Stack:**
    - LangChain + ChromaDB
    - sentence-transformers embeddings
    - Mistral 7B via Ollama
    - RAGAS evaluation
    """)

    st.divider()
    st.header("Loaded documents")
    try:
        from src.embedder import load_vector_store
        vs = load_vector_store()
        count = vs._collection.count()
        st.success(f"{count} chunks indexed")

        raw_files = list(Path("data/raw").iterdir())
        if raw_files:
            st.caption("Documents:")
            for f in raw_files:
                st.caption(f"📄 {f.name}")
    except Exception:
        st.warning("No documents indexed yet. Upload some above!")

    st.divider()
    st.header("Try asking")
    st.markdown("""
    - What is federated learning?
    - What is Grad-CAM?
    - What ethical challenges does AI face?
    - What is the EchoNet dataset?
    - Summarise this document
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