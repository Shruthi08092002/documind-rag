import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
# This is where we tell the ingestor where to find documents
# and how to split them
RAW_DATA_PATH = Path("data/raw")
CHUNK_SIZE = 500      # Each chunk will be ~500 characters
CHUNK_OVERLAP = 50    # Chunks overlap by 50 chars so we don't lose context at edges

def load_documents(data_path: Path) -> list:
    """
    Walks through the data/raw folder and loads all
    PDF and text files it finds.
    Returns a list of LangChain Document objects.
    """
    documents = []
    supported_extensions = {".pdf", ".txt", ".md"}

    print(f"\n📂 Scanning folder: {data_path}")

    for file_path in data_path.iterdir():
        # Skip files we can't handle
        if file_path.suffix.lower() not in supported_extensions:
            print(f" Skipping unsupported file: {file_path.name}")
            continue

        print(f"  📄 Loading: {file_path.name}")

        try:
            if file_path.suffix.lower() == ".pdf":
                # PyMuPDF handles PDFs — extracts text page by page
                loader = PyMuPDFLoader(str(file_path))
            else:
                # TextLoader handles .txt and .md files
                loader = TextLoader(str(file_path), encoding="utf-8")

            docs = loader.load()

            # Tag each document with its source filename
            # This is metadata — we'll use it later to show citations
            for doc in docs:
                doc.metadata["source"] = file_path.name

            documents.extend(docs)
            print(f"   Loaded {len(docs)} page(s)")

        except Exception as e:
            print(f"   Error loading {file_path.name}: {e}")

    print(f"\n📊 Total pages/sections loaded: {len(documents)}")
    return documents


def split_documents(documents: list) -> list:
    """
    Takes the loaded documents and splits them into
    smaller chunks using RecursiveCharacterTextSplitter.

    Why Recursive? It tries to split on natural boundaries:
    first paragraphs, then sentences, then words.
    This keeps chunks meaningful rather than cutting mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunks")
    print(f"    (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Show a preview of the first chunk so we know it's working
    if chunks:
        print(f"\n🔍 Preview of first chunk:")
        print(f"   Source: {chunks[0].metadata.get('source', 'unknown')}")
        print(f"   Text: {chunks[0].page_content[:200]}...")

    return chunks


def ingest(data_path: Path = RAW_DATA_PATH) -> list:
    """
    Main function that runs the full ingestion pipeline:
    Load → Split → Return chunks
    """
    if not data_path.exists():
        print(f"Data folder not found: {data_path}")
        return []

    files = list(data_path.iterdir())
    if not files:
        print(f"No files found in {data_path}")
        print("Add some PDF or text files to data/raw/ and try again")
        return []

    # Step 1: Load all documents
    documents = load_documents(data_path)

    if not documents:
        return []

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    return chunks


# This block runs only when you execute this file directly
# i.e. python src/ingestor.py
if __name__ == "__main__":
    chunks = ingest()
    print(f"\n Ingestion complete! {len(chunks)} chunks ready for embedding.")