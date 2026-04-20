# 🧠 DocuMind — Local RAG Pipeline with Evaluation

A fully local, end-to-end Retrieval-Augmented Generation (RAG) system that ingests documents, indexes them into a vector store, answers questions via a chat UI, and evaluates pipeline quality using automated scoring.

Built with zero API costs — everything runs on your machine.


## Demo

> Upload any PDF → Ask questions in plain English → Get grounded answers with source citations


## What it does

- Ingests PDF, TXT, and Markdown files
- Chunks documents using recursive character splitting
- Embeds chunks into ChromaDB using sentence-transformers
- Retrieves semantically relevant chunks per question
- Generates grounded answers via Mistral 7B (Ollama)
- Evaluates pipeline quality across faithfulness, answer relevancy, and context recall
- Serves a Streamlit chat UI with source citations and document upload


## Architecture

PDF/TXT → Ingestor → Chunker → Embedder → ChromaDB
↓
User Question → Retriever → Generator (Mistral) → Answer + Sources


## Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Mistral 7B via Ollama |
| Evaluation | Custom scoring (faithfulness, relevancy, recall) |
| UI | Streamlit |
| Language | Python 3.12 |



## Evaluation Results

Evaluated across 20 questions with ground truth answers from an academic paper on AI in medical imaging.

| Config | Faithfulness | Answer Relevancy | Context Recall |
|---|---|---|---|
| baseline_k3 | 0.854 | 0.435 | 0.507 |
| experiment_k5 | **0.900** | **0.442** | **0.543** |
| experiment_mmr | 0.776 | 0.442 | 0.376 |

**Key finding:** Increasing retrieved chunks from k=3 to k=5 improved context recall by 7.1% and faithfulness by 5.4%. MMR retrieval underperformed on this dense academic corpus because diversity hurt more than it helped.



## Project Structure

```
documind-rag/
├── src/
│   ├── ingestor.py         # Document loading and chunking
│   ├── embedder.py         # Embedding and ChromaDB storage
│   ├── retriever.py        # Semantic search over vector store
│   └── generator.py        # LLM answer generation
├── tests/
│   ├── test_set.py         # 20-question evaluation set
│   ├── evaluator.py        # Scoring pipeline
│   └── run_experiments.py  # Multi-config comparison
├── data/
│   └── raw/                # Drop documents here
├── app.py                  # Streamlit chat UI
└── requirements.txt
```




## Setup

**Prerequisites:** Python 3.10+, Ollama installed

**1. Clone the repo**
```bash
git clone https://github.com/Shruthi08092002/documind-rag.git
cd documind-rag
```

**2. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Pull Mistral via Ollama**
```bash
ollama pull mistral
```

**4. Add documents and embed**
```bash
python -m src.embedder
```

**5. Run the app**
```bash
# Terminal 1
ollama serve

# Terminal 2
streamlit run app.py
```

---

## Running Evaluation

```bash
# Baseline evaluation
python -m tests.evaluator

# Run all experiments and compare
python -m tests.run_experiments
```



## Key Design Decisions

**Why local?** No API costs, no data privacy concerns, works offline. Ideal for sensitive documents.

**Why ChromaDB?** Persistent, lightweight, runs on CPU. No external service needed.

**Why all-MiniLM-L6-v2?** 80MB, fast on CPU, strong semantic search performance for its size.

**Why k=5?** Evaluation showed 7.1% recall improvement over k=3 with no faithfulness penalty.


## Limitations

- Structural queries ("what is on page 3?") work poorly — chunking destroys document structure
- Mistral 7B on CPU generates answers in 30-60 seconds
- RAGAS parallel scoring requires a faster LLM for evaluation


## Author

Shruthi Srinivasan 