# AI Paralegal Assistant

An AI-powered legal assistant that helps with document analysis, legal Q&A, and legal document generation.

## Features

- **Document Q&A**: Ask questions about legal documents stored in the system
- **Legal Document Generation**: Generate various legal documents (writ petitions, affidavits, etc.)
- **User-friendly Interface**: Easy-to-use Streamlit interface

## Installation

1. Clone this repository
2. Make sure you have Python 3.8+ installed
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your environment variables by creating a `.env` file with:

```
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

## Running the Application

You can run the application in two ways:

### Option 1: Run both servers at once (recommended)

```bash
python run_app.py
```

This will start both the FastAPI backend server and the Streamlit frontend.

### Option 2: Run servers separately

In one terminal, start the backend:

```bash
python main.py
```

In another terminal, start the Streamlit frontend:

```bash
streamlit run streamlit_app.py
```

## Usage

1. Open your browser and go to http://localhost:8501
2. Use the sidebar to navigate between the Chat and Document Generation features
3. In Chat mode, ask questions about legal documents stored in the system
4. In Document Generation mode, select the type of document you want to create and fill in the required information

## Project Structure

- `main.py`: FastAPI backend for document generation
- `rag_chatbot.py`: RAG (Retrieval-Augmented Generation) chatbot for legal Q&A
- `doc_draft.py`: Document generation functions
- `streamlit_app.py`: Streamlit frontend
- `embeddings.py`: Embeddings generation for document indexing
- `run_app.py`: Helper script to run both servers 