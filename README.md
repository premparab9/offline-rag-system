# Offline RAG System

A local Retrieval-Augmented Generation (RAG) system that allows you to upload documents and ask questions based on their content without using any external APIs.

The entire pipeline runs offline using local models.

---

## Features

* Upload documents (PDF, DOCX, TXT)
* Ask questions based on uploaded content
* Answers generated using local LLM
* Works completely offline
* Source chunks shown for transparency

---

## How it works

1. Documents are loaded and text is extracted
2. Text is cleaned and split into chunks
3. Each chunk is converted into embeddings
4. Stored in a vector database (ChromaDB)
5. On query:

   * Relevant chunks are retrieved
   * Passed to a local LLM (Ollama)
   * Answer is generated using only retrieved context

---

## Tech Stack

* Python
* Streamlit
* LangChain
* Ollama (LLM + embeddings)
* ChromaDB

---

## Project Structure

```id="7cz4b6"
app.py              # UI
ingest.py           # Document processing
query.py            # Q&A logic
llm.py              # LLM integration
embeddings.py       # Embedding generation
logger.py
vector_store.py     # Database operations
requirements.txt
config.py           # Settings
utils/              # File loaders and cleaning
docx_loader.py
ocr_loader.py
pdf_loader.py
text_clearer.py
```

---

## Setup Instructions

Follow the steps below to run the project locally.

---

### 1. Clone the repository

```
git clone https://github.com/premparab9/offline-rag-system.git
cd offline-rag-system
```

---

### 2. Create and activate virtual environment

```
python -m venv .venv
```

Activate the environment:

* **Windows (PowerShell):**

```
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
---

### 4. Install and start Ollama

Download and install Ollama from: https://ollama.com

Start the Ollama server:

```bash
ollama serve
```

---

### 5. Pull required models

```bash
ollama pull gemma2:2b
ollama pull nomic-embed-text
```

---

### 6. Run the application

```
streamlit run app.py
```

---

### 7. Open in browser

The app will be available at:

```
http://localhost:8501
```

---

## Notes

* First-time model download may take time depending on internet speed
* Make sure Ollama is running before starting the app
* For image support (OCR), install Tesseract separately

## Usage

* Upload documents
* Click **Ingest Documents**
* Ask questions
* View answers with source context

---

## Author

Prem Parab
