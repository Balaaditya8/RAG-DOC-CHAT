# RAG-DOC-Chat

## Overview

RAG-DOC-Chat is a Streamlit-based application that allows users to upload a PDF and ask questions about its content using a Retrieval-Augmented Generation (RAG) approach. The app uses ChromaDB for vector storage and integrates with the Groq API for large language model-based responses.

## Features

- 📄 Upload and process PDFs
- 🔍 Retrieve context-aware answers from the document
- 💬 Interactive chat format with history
- 🧠 Uses HuggingFace embeddings and ChromaDB for retrieval
- 🤖 Integrates with the Groq API for LLM responses

## Installation

To run this project locally, follow these steps:

### Prerequisites

- Python 3.8+
- pip package manager
- A Groq API key

### Setup Instructions

1. **Clone the repository:**

   ```sh
   git clone https://github.com/Balaaditya8/RAG-DOC-Chat.git
   cd RAG-DOC-Chat
   ```

2. **Create a virtual environment (optional but recommended):**

   ```sh
   python -m venv venv
   source venv/bin/activate  
   ```

3. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Run the application:**

   ```sh
   streamlit run app.py
   ```

## Usage

1. Start the application by running `streamlit run app.py`.
2. Upload a PDF file.
3. Enter your Groq API key when prompted.
4. Ask questions related to the document.
5. View responses and chat history.

## Folder Structure

```
RAG-DOC-Chat/
│-- app.py              # Main Streamlit app
│-- requirements.txt    # List of dependencies
│-- README.md           # Project documentation
│-- chroma_db/          # Stores vectorized document chunks
```

## Technologies Used

- **Streamlit** - Web framework for interactive applications
- **ChromaDB** - Vector database for efficient document retrieval
- **LangChain** - Framework for building LLM applications
- **HuggingFace Transformers** - Embeddings for text processing
- **Groq API** - Large Language Model integration

---

Enjoy using **RAG-DOC-Chat**! 🚀

