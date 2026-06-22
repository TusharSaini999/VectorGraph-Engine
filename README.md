# 🌌 VectorGraph-Engine

**VectorGraph-Engine** is a powerful **local-first Retrieval-Augmented Generation (RAG) pipeline** and **automated Knowledge Graph extraction engine**. Built with an interactive Streamlit interface, it processes complex PDF documents (including scanned documents via OCR), converts them into semantic vector embeddings using local HuggingFace models, and leverages Google Gemini to automatically extract relationships and construct interactive Knowledge Graphs.

---


## ✨ Features

### 📄 Hybrid Document Processing

* Extracts text from standard PDF documents.
* Automatically falls back to OCR using Tesseract for scanned PDFs and image-based pages.
* Handles mixed-content documents seamlessly.

### 🧩 Semantic Chunking & Metadata Enrichment

* Intelligent text segmentation using LangChain recursive text splitters.
* Preserves document structure with section-aware chunking.
* Injects source metadata directly into chunks for improved retrieval quality.

### 🧠 Local Vector Embeddings

* Privacy-focused embedding generation using:

  * `sentence-transformers/all-MiniLM-L6-v2`
* Runs locally without requiring external embedding APIs.
* Generates lightweight 384-dimensional semantic vectors.

### ⚡ High-Speed Semantic Search

* Powered by FAISS (Facebook AI Similarity Search).
* Fast vector indexing and nearest-neighbor retrieval.
* Supports semantic document exploration and question answering.

### 🧬 Automated Knowledge Graph Generation

* Uses **Google Gemini 3.5 Flash** to analyze document chunks.

* Extracts semantic triples:

  ```
  Subject → Predicate → Object
  ```

* Automatically builds structured knowledge representations from unstructured text.

### 📊 Interactive Analytics & Visualization

* PCA-based vector space visualization using Plotly.
* Interactive Knowledge Graph rendering with PyVis.
* Physics-based graph navigation and exploration.

### 🔄 Resilient Processing Pipeline

* Auto-saves progress during indexing.
* Resume interrupted vectorization processes.
* Continue Knowledge Graph generation from previous checkpoints.

---

# 🏗️ System Architecture

```text
PDF Documents
      │
      ▼
Document Parser
(PyPDF + OCR)
      │
      ▼
Text Chunking
(LangChain)
      │
      ▼
Embeddings
(MiniLM-L6-v2)
      │
      ▼
FAISS Vector Store
      │
      ├────────► Semantic Search
      │
      ▼
Gemini 2.5 Flash
      │
      ▼
Knowledge Graph Extraction
      │
      ▼
PyVis Visualization
```

---

# 🛠️ Tech Stack

| Category        | Technologies                      |
| --------------- | --------------------------------- |
| Frontend        | Streamlit, Plotly, PyVis          |
| AI Models       | HuggingFace Sentence Transformers |
| LLM             | Google Gemini 2.5 Flash           |
| Framework       | LangChain                         |
| Vector Database | FAISS                             |
| OCR             | Tesseract OCR                     |
| PDF Processing  | PyPDF, pdf2image                  |
| Analytics       | NumPy, Pandas, Scikit-Learn       |

---

# ⚙️ Prerequisites

Before installing Python dependencies, install the following system-level tools.

## 1. Tesseract OCR

### Windows

Download from:
https://github.com/UB-Mannheim/tesseract/wiki

### Linux

```bash
sudo apt-get install tesseract-ocr
```

### macOS

```bash
brew install tesseract
```

---

## 2. Poppler

Required for PDF-to-image conversion.

### Windows

Download the latest release and extract it. Note the path to the `Library/bin` directory.

### Linux

```bash
sudo apt-get install poppler-utils
```

### macOS

```bash
brew install poppler
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/VectorGraph-Engine.git
cd VectorGraph-Engine
```

## Create Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages include:

```txt
streamlit
langchain
langchain-huggingface
langchain-google-genai
sentence-transformers
faiss-cpu
numpy
pandas
plotly
pytesseract
pdf2image
pypdf
scikit-learn
pyvis
python-dotenv
```

---

# 🔐 Environment Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional (Windows)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Optional (Windows)
POPPLER_PATH=C:\path\to\poppler\Library\bin
```

---

# ▶️ Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# 📖 Dashboard Overview

## 🏃 Pipeline

* Upload PDF documents.
* Configure chunking parameters.
* Generate embeddings and build the FAISS index.

---

## 🔎 Search

* Test semantic retrieval.
* Query the vector database using natural language.

---

## 📚 Inspector

* Explore processed chunks.
* Filter by source document.
* Verify metadata and chunk quality.

---

## 📉 Analytics

* View vector database statistics.
* Generate PCA visualizations of embedding clusters.

---

## 📝 History Log

* Monitor OCR operations.
* Track indexing progress.
* Debug processing errors.

---

## 🧬 KG Generator

* Send vectorized chunks to Gemini.
* Extract semantic triples automatically.
* Generate Knowledge Graph datasets.

---

## 🌐 KG Visualizer
<img width="616" height="503" alt="Screenshot 2026-06-22 123635" src="https://github.com/user-attachments/assets/3d851928-ac73-41d7-9b91-44272036c088" />

* Explore graph relationships interactively.
* Search entities and connections.
* Analyze document knowledge structures.

---

# 🎯 Use Cases

* Research Paper Analysis
* Enterprise Knowledge Management
* Legal Document Exploration
* Technical Documentation Search
* Academic Literature Review
* Automated Knowledge Graph Construction
* Local AI-Powered Document Intelligence

---

# 🔒 Privacy First

VectorGraph-Engine is designed with a **local-first architecture**:

✅ Local embedding generation

✅ Local FAISS storage

✅ No cloud vector databases required

✅ Offline semantic search

⚠️ Only Knowledge Graph extraction requires Gemini API access.

---

# 🛡️ License

Distributed under the **GPL-3.0 License**.

See the `LICENSE` file for more information.

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

# ⭐ Support the Project

If you find VectorGraph-Engine useful, consider giving the repository a star ⭐ on GitHub to support development and future improvements.
