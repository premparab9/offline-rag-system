# 🔒 Offline RAG System

A local Retrieval-Augmented Generation (RAG) system that allows you to upload documents and ask questions based on their content — without using any external APIs.

The entire pipeline runs offline using local models.

---

## 🚀 Features

* Upload documents (PDF, DOCX, TXT, Images)
* Ask questions based on uploaded content
* Answers generated using local LLM
* Works completely offline
* Source chunks shown for transparency

---

## 🧠 How it works

1. Documents are loaded and text is extracted
2. Text is cleaned and split into chunks
3. Each chunk is converted into embeddings
4. Stored in a vector database (ChromaDB)
5. On query:

   * Relevant chunks are retrieved
   * Passed to a local LLM (Ollama)
   * Answer is generated using only retrieved context

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* Ollama (LLM + embeddings)
* ChromaDB

---

## 📂 Project Structure

```id="7cz4b6"
app.py              # UI
ingest.py           # Document processing
query.py            # Q&A logic
llm.py              # LLM integration
embeddings.py       # Embedding generation
vector_store.py     # Database operations
config.py           # Settings
utils/              # File loaders and cleaning
```

## 💡 Usage

* Upload documents
* Click **Ingest Documents**
* Ask questions
* View answers with source context

---

## 👨‍💻 Author

Prem Parab
