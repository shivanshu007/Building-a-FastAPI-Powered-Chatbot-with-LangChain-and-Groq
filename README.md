# FastAPI-Powered Chatbot with LangChain and Groq

This project implements a chatbot using FastAPI, LangChain, Groq, and FAISS for document-based question answering. It retrieves content from a web source, processes it into chunks, and uses a language model to provide accurate responses based on the context.

## Features

- Document-based question answering system.
- LangChain integration for document retrieval and processing.
- Embeddings generated with Ollama and stored using FAISS for vector search.
- CORS enabled to allow API access from any domain.
- Reloadable document embeddings via API for real-time updates.

## Prerequisites

- Python 3.9+
- Groq API Key (store in `.env` file).
- Required Python packages: `FastAPI`, `uvicorn`, `langchain_groq`, `faiss-cpu`, `ollama`, `pydantic`, `python-dotenv`.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/chatbot-fastapi-langchain.git
    cd chatbot-fastapi-langchain
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Add your Groq API key in the `.env` file:**
    ```
    GROQ_API_KEY=<your_groq_api_key>
    ```

## Usage

1. **Start the FastAPI server:**
    ```bash
    uvicorn main:app --reload
    ```

2. **Available API Endpoints:**

   - **POST `/chat`:** Query the chatbot with a prompt and receive a response based on document context.
   - **GET `/documents`:** Retrieve chunks of the loaded documents for exploration or debugging.
   - **POST `/reload_documents`:** Reload documents and update embeddings dynamically.

## API Documentation

### 1. Chat API

- **Endpoint:** `/chat`
- **Method:** `POST`
- **Request Body:**

  ```json
  {
    "prompt": "What is LangChain?"
  }


Response:

  ```bash
   {
      "answer": "LangChain is a framework for developing applications...",
      "context": ["LangChain is a...", "LangChain uses..."]
    }
  ```

2. Documents API
Endpoint: `/documents`

Method: `GET`

Response: Returns a list of document chunks.
```bash
[
  "LangChain is a framework for...",
  "The primary components of LangChain include..."
]
```
3. Reload Documents API
Endpoint: `/reload_documents`

Method: `POST`

Response: Reloads the documents and updates the embeddings.
```bash
{
  "status": "Documents reloaded successfully"
}
```
Project Structure
```bash
.
├── main.py              # FastAPI application
├── requirements.txt     # List of dependencies
├── .env                 # Environment variables (e.g., API keys)
└── README.md            # Project documentation
```
## Key Components
LangChain Groq: Utilizes the Groq language model for question-answering.
FAISS Vector Store: Stores and retrieves vectorized document chunks for efficient search.
WebBaseLoader: Loads documents from the provided URL.
FastAPI: Provides an API for interacting with the chatbot.
Sample Queries
You can try asking the following questions:
```bash
"What is LangChain?"
"How does FAISS work?"
"Explain the key components of LangChain."
```
Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests for bug fixes, feature enhancements, or improvements.


License
This project is licensed under the MIT License.
