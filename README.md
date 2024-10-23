# Building-a-FastAPI-Powered-Chatbot-with-LangChain-and-Groq
In this article, we will walk through the steps to create a chatbot using FastAPI, LangChain, and Groq. This chatbot is designed to retrieve documents from a web source, split the content into manageable chunks, and use FAISS for vectorized document retrieval. We will also integrate the chatbot with LangChain's document chain and Groq’s language models for handling natural language queries.

Prerequisites
Before you begin, you should have:

Basic knowledge of Python
Familiarity with FastAPI
Understanding of LangChain and FAISS
An installed Python environment with FastAPI, uvicorn, and related packages
Setting Up the Project
1. Install Required Dependencies
Start by installing the necessary packages for this project:

bash
Copy code
pip install fastapi uvicorn langchain_groq faiss-cpu langchain-embeddings ollama pydantic langchain-community
Additionally, you will need to install the python-dotenv package to load environment variables:

bash
Copy code
pip install python-dotenv
2. Setting Up the FastAPI App
We will start by importing the required modules and creating a simple FastAPI application.

python
Copy code
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import time
3. Setting Up Environment Variables
We’ll be using Groq’s API, so you’ll need to store the API key in an .env file. Create a .env file in your project directory with the following content:

makefile
Copy code
GROQ_API_KEY=<your_groq_api_key>
In the code, use the dotenv library to load the API key.

python
Copy code
# Load environment variables
load_dotenv()

# Load the Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
4. Initializing the Chatbot Components
a. Document Loader
We use WebBaseLoader to load documents from a web URL. Here, we load content from the LangChain documentation site as an example.

python
Copy code
# Load documents from a website
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
b. Text Splitter
Once the documents are loaded, the content is split into smaller chunks for more efficient retrieval using RecursiveCharacterTextSplitter.

python
Copy code
# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs[:50])
c. Embeddings and FAISS Vector Store
We use Ollama embeddings to convert document chunks into vectors and store them in a FAISS vector store for retrieval.

python
Copy code
# Convert documents to vectors using embeddings
embeddings = OllamaEmbeddings()
vectors = FAISS.from_documents(final_documents, embeddings)
d. Language Model (LLM)
For handling natural language queries, we initialize the Groq model via the ChatGroq class. You can specify the model you wish to use, such as mixtral-8x7b-32768.

python
Copy code
# Initialize the language model with Groq API
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
e. Prompt Template and Chain
Next, we create a prompt template and document chain for the chatbot. This chain will be used to retrieve and answer questions based on the provided document context.

python
Copy code
# Define the prompt template
prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
document_chain = create_stuff_documents_chain(llm, prompt)
f. Retrieval Chain
The vectorized documents are passed to the retrieval_chain for answering user queries.

python
Copy code
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
5. Defining the API Endpoints
a. Chat Endpoint
This endpoint accepts a user query (prompt) and returns an answer from the chatbot, along with the context used for the answer.

python
Copy code
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    answer: str
    context: List[str]

@app.post("/chat", response_model=ChatResponse)
def get_response(request: ChatRequest):
    """
    Chat endpoint for generating a response based on the prompt and context.
    """
    try:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": request.prompt})
        print("Response time: ", time.process_time() - start)

        return ChatResponse(
            answer=response["answer"],
            context=[doc.page_content for doc in response["context"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
b. Documents Endpoint
This endpoint returns the first 10 chunks of the loaded documents for debugging or exploration purposes.

python
Copy code
@app.get("/documents", response_model=List[str])
def get_documents():
    """
    Endpoint to get loaded document chunks (for debugging or exploration).
    """
    return [doc.page_content for doc in final_documents[:10]]
c. Reload Documents Endpoint
To refresh the document chunks and vectors in real-time, this endpoint reloads the documents and updates the embeddings.

python
Copy code
@app.post("/reload_documents")
def reload_documents():
    """
    Endpoint to reload the documents and embeddings.
    """
    global docs, final_documents, vectors, retriever

    try:
        docs = loader.load()
        final_documents = text_splitter.split_documents(docs)
        vectors = FAISS.from_documents(final_documents, embeddings)
        retriever = vectors.as_retriever()

        return {"status": "Documents reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
6. Running the FastAPI App
Finally, use uvicorn to run the FastAPI app.

bash
Copy code
uvicorn main:app --reload
Conclusion
In this article, we created a chatbot that can respond to queries based on documents loaded from a web source. By integrating LangChain’s document chain and FAISS vector retrieval, we built a robust system for document-based question answering. You can expand this project by integrating additional document sources, fine-tuning the models, or adjusting the prompt template for different use cases.
