# 🧠 DocMind — Intelligent Knowledge Retrieval & Generation System

<p align="center">
  <img src="https://img.shields.io/badge/Built%20By-Umair%20Rahman%20Shaik-6c63ff?style=for-the-badge&logo=github"/>
  <img src="https://img.shields.io/badge/AI%20Framework-txtai-00b4d8?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/UI-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python"/>
</p>

---

## 📌 Overview

**DocMind** is an advanced AI-powered knowledge retrieval and generation system that combines **semantic vector search** with **large language model (LLM) reasoning** to enable intelligent Q&A over any custom document corpus.

Unlike traditional keyword search, DocMind understands the *semantic meaning* behind queries — finding conceptually relevant information even when exact keywords don't match. It integrates a dual-retrieval architecture:

- **Vector RAG** — Dense embedding-based retrieval using transformer models
- **Graph RAG** — Knowledge graph traversal using semantic relationships between entities

> 💡 **Core Idea**: Constrain what the LLM "knows" to only your documents — eliminating hallucinations and making answers factually grounded.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DocMind Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Query ──► [Intent Classifier]                         │
│                       │                                     │
│           ┌───────────┴───────────┐                         │
│           ▼                       ▼                         │
│     Vector Search           Graph Traversal                 │
│   (Dense Retrieval)       (Cypher Path Query)               │
│           │                       │                         │
│           └───────────┬───────────┘                         │
│                       ▼                                     │
│               Context Assembly                              │
│                       │                                     │
│                       ▼                                     │
│            LLM Prompt Construction                          │
│                       │                                     │
│                       ▼                                     │
│           Grounded Natural Language Answer                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Semantic Search** | Dense vector retrieval using transformer embeddings (e5-large) |
| 📈 **Graph RAG** | Knowledge graph-based context expansion using semantic relationships |
| 📄 **Multi-format Ingestion** | Index PDFs, web URLs, Word docs, plain text, and more |
| 🤖 **LLM-agnostic** | Works with local models (Qwen, LLaMA) or cloud APIs (GPT, Claude) |
| 💬 **Conversational UI** | Real-time streaming chat interface powered by Streamlit |
| ⚡ **Offline-capable** | Fully local execution — no data leaves your machine |
| 🔁 **Dynamic Indexing** | Add new documents to the knowledge base without restarting |
| 🗺️ **Graph Visualization** | Renders knowledge graph diagrams for Graph RAG queries |

---

## 🧬 Technical Stack

| Layer | Technology |
|-------|-----------|
| **Embedding Model** | `intfloat/e5-large` (1024-dim dense retrieval) |
| **Vector Store** | `txtai Embeddings` (FAISS-backed ANN index) |
| **Knowledge Graph** | `txtai Graph` + NetworkX + Grand-Cypher (Cypher query engine) |
| **LLM Runtime** | `txtai LLM` (HuggingFace Transformers / llama.cpp / API) |
| **RAG Pipeline** | `txtai RAG` (templated prompt construction + context injection) |
| **Text Extraction** | `txtai Textractor` (multi-format document parser) |
| **Frontend** | `Streamlit` (real-time streaming chat) |
| **Language** | Python 3.10+ |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/URS05/docmind.git
cd docmind
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure (Optional)

Copy the example config and edit:

```bash
cp .env.example .env
```

Edit `.env` to set your LLM, embeddings path, etc.

### 4. Launch DocMind

```bash
streamlit run rag.py
```

Open your browser at **http://localhost:8501**

---

## 📖 Usage Guide

### Querying the Knowledge Base

| Query Type | Example | Description |
|------------|---------|-------------|
| **Standard** | `What is machine learning?` | Vector similarity search + LLM answer |
| **Graph RAG** | `gq: Explain neural networks` | Graph-expanded context + LLM answer |
| **Path Query** | `CNN -> RNN -> Transformer` | Traverses concept graph path |
| **Combined** | `CNN -> Transformer gq: Compare architectures` | Path + graph query |

### Adding Documents to the Index

Prefix your message with `#` to ingest content:

```
# https://arxiv.org/abs/1706.03762        ← Index a research paper
# /path/to/your/document.pdf              ← Index a local file
# Custom text note: Neural networks are...  ← Index raw text
```

### Check System Settings

```
:settings
```

---

## ⚙️ Configuration Reference

All settings can be set via environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `TITLE` | Application title | `DocMind` |
| `LLM` | LLM model path or API model name | `Qwen/Qwen2.5-0.5B-Instruct` |
| `EMBEDDINGS` | Pre-built embeddings index path | *(empty — starts fresh)* |
| `CONTEXT` | Number of context chunks retrieved | `10` |
| `MAXLENGTH` | Max generation token length | `2048` |
| `STRIPTHINK` | Strip chain-of-thought tokens | `False` |
| `DATA` | Directory of files to auto-index at startup | `None` |
| `PERSIST` | Directory to persist index updates | `None` |

### Example: Use with OpenAI GPT-4

```bash
LLM=gpt-4o OPENAI_API_KEY=sk-... streamlit run rag.py
```

### Example: Use with Ollama (Local)

```bash
LLM=ollama/llama3 streamlit run rag.py
```

### Example: Auto-index a folder of documents

```bash
DATA=./my_documents PERSIST=./index streamlit run rag.py
```

---

## 🔬 How RAG Works

### Vector RAG Pipeline

```
Query → Embed (e5-large) → ANN Search (FAISS) → Top-K Chunks
     → Inject into Prompt → LLM → Grounded Answer
```

### Graph RAG Pipeline

```
Query → Parse Graph Intent → Cypher Path Query (Grand-Cypher)
     → Graph Traversal (NetworkX) → Related Nodes
     → Context Assembly → LLM → Rich Summarized Answer
     → Visualize Graph (matplotlib)
```

---

## 📁 Project Structure

```
docmind/
├── rag.py              # Core application — RAG engine + Streamlit UI
├── requirements.txt    # Python dependencies
├── .env.example        # Configuration template
├── Dockerfile          # Docker deployment config
└── README.md           # This file
```

---

## 📜 Author

**Umair Rahman Shaik**  
GitHub: [@URS05](https://github.com/URS05)

---

## 📄 License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
