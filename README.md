# Multi-PDF Conversational AI (RAG-based)

A Retrieval-Augmented Generation (RAG) application that enables users to upload multiple PDF documents and interact with them using natural language queries. The system retrieves relevant document chunks and generates context-aware responses using a transformer-based language model.

---

## Features

- Runs Locally and Offline
- Upload and process multiple PDF documents
- Semantic search using vector embeddings (FAISS)
- Conversational question-answering with memory
- Context-aware responses using a transformer-based LLM (Flan-T5)
- Efficient text chunking for improved retrieval accuracy
- Real-time interaction via Streamlit interface

---
## Demo Video
https://github.com/user-attachments/assets/fc3b6153-cb93-4567-adb3-d36cf4f73c02

## System Architecture

### 1. PDF Ingestion
- Extracts raw text from PDFs using PyPDF2

### 2. Text Chunking
- Splits large text into smaller overlapping chunks for better retrieval

### 3. Embedding Generation
- Converts text chunks into dense vector representations using Sentence Transformers

### 4. Vector Storage
- Stores embeddings in FAISS for efficient similarity search

### 5. Retrieval-Augmented Generation (RAG)
- Retrieves top-k relevant chunks
- Passes them to the LLM (Flan-T5) to generate answers grounded in context

### 6. Conversational Memory
- Maintains chat history to support context-aware multi-turn conversations

---

## Tech Stack

| Component        | Technology Used                          |
|----------------|------------------------------------------|
| Frontend        | Streamlit                                |
| LLM             | Hugging Face Transformers (Flan-T5)      |
| Embeddings      | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS                                    |
| Framework       | LangChain                                |
| PDF Processing  | PyPDF2                                   |

---

## Project Structure

├── app.py # Main Streamlit application
├── htmlTemplates.py # Chat UI templates and styling
├── requirements.txt # Project dependencies
├── README.md # Project documentation


---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/multi-pdf-chat.git
cd multi-pdf-chat
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
```
## Usage
- Upload one or more PDF files using the sidebar
- Click on "Process" to extract and index content
- Enter your query in the input box
- Receive answers based on document context
## Example Use Cases
- Research paper analysis
- Legal document exploration
- Business report querying
- Academic study assistant
## Limitations
- May generate incorrect answers if relevant context is not retrieved
- Performance depends on chunking strategy and embedding quality
- CPU-based inference leads to slower response times
- Not optimized for very large-scale document collections
## Future Improvements
- Integrate stronger LLMs (Gemini, Mistral API)
- Add cross-encoder reranking for improved retrieval precision
- Improve grounding to reduce hallucinations
- Enable deployment on cloud platforms (AWS, Streamlit Cloud)
- Add source citations in generated responses





