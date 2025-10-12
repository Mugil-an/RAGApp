# RAG Application with Streamlit, Inngest, and Gemini

This project is a complete Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content. It uses an asynchronous, event-driven architecture powered by Inngest to handle document processing and querying, ensuring the UI remains responsive.

The frontend is built with Streamlit, providing a simple, interactive web interface. The backend is a FastAPI application that serves the Inngest functions. Google's Gemini model is used for generating text embeddings and synthesizing answers, and Qdrant is used as the vector database for efficient similarity searches.

## Features

-   **PDF Document Upload**: Simple interface to upload and ingest PDF files.
-   **Asynchronous Processing**: Uses Inngest to manage long-running tasks like document chunking, embedding, and upserting without blocking the UI.
-   **Vector-Based Retrieval**: Creates embeddings from text chunks and stores them in a Qdrant vector database.
-   **Generative Question-Answering**: Uses Google's Gemini model to generate answers based on the context retrieved from the vector database.
-   **Interactive Chat UI**: A user-friendly chat interface for asking questions and viewing answers and their sources.
-   **Tabbed Interface**: Cleanly separates the document upload and querying functionalities.

## Architecture

The application consists of three main components:

1.  **Streamlit Frontend (`streamlit_app.py`)**: The user-facing application where users can upload PDFs and ask questions. It communicates with the Inngest backend by sending events.
2.  **FastAPI & Inngest Backend (`main.py`)**: A FastAPI server that hosts the Inngest functions. These functions define the multi-step workflows for:
    -   **Ingesting a PDF**: `rag/inngest_pdf` event triggers a function that loads, chunks, embeds, and stores the document content in Qdrant.
    -   **Querying**: `rag/query` event triggers a function that embeds the user's question, searches Qdrant for relevant context, and then passes that context to the Gemini model to generate a final answer.
3.  **Services**:
    -   **Qdrant**: A vector database used to store the document embeddings for fast and scalable similarity search.
    -   **Google Gemini**: The LLM used for both creating embeddings and generating answers.
    -   **Inngest Dev Server**: A local server for developing and testing Inngest functions.

## Prerequisites

-   Python 3.8+
-   Docker and Docker Compose
-   A Google AI Studio API Key

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Mugil-an/RAGApp.git
    cd RAGApp
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a file named `.env` in the root directory and add your Google API key:
    ```env
    # .env
    GOOGLE_API_KEY="your_google_api_key_here"
    INNGEST_API_BASE_URL="http://127.0.0.1:8288/v1"
    ```

5.  **Start dependent services (Qdrant & Inngest):**
    You will need a `docker-compose.yml` file to run Qdrant and the Inngest dev server.
    ```yaml
    # docker-compose.yml
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - "6333:6333"
        volumes:
          - ./qdrant_data:/qdrant/storage

      inngest-dev:
        image: inngest/dev-server:latest
        ports:
          - "8288:8288" # API
          - "4321:4321" # UI
    ```
    Run Docker Compose from your terminal:
    ```bash
    docker-compose up
    ```

## Running the Application

You need to run two processes in separate terminals.

1.  **Run the FastAPI/Inngest Backend:**
    ```bash
    uvicorn main:app --reload
    ```

2.  **Run the Streamlit Frontend:**
    ```bash
    streamlit run streamlit_app.py
    ```

Now you can access:
-   The Streamlit App at `http://localhost:8501`
-   The Inngest Dev Server UI at `http://localhost:4321`

## How to Use

1.  Open your browser and navigate to `http://localhost:8501`.
2.  On the **"Upload Document"** tab, choose a PDF file to upload.
3.  Wait for the success message confirming that the ingestion process has started.
4.  Switch to the **"Query Documents"** tab.
5.  Type your question into the chat input at the bottom and press Enter.
6.  The assistant will process your question and return an answer along with the sources it used.