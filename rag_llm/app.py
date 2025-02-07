from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from uuid import uuid4
import os
from dotenv import load_dotenv

# Load environment variables
print("[INFO] Loading environment variables...")
load_dotenv()

# Set paths for storing documents and Chroma vector store
DATA_PATH = r"data"
CHROMA_PATH = "./chroma_db"

# Initialize FastAPI application
print("[INFO] Initializing FastAPI application...")
app = FastAPI()

# Load the pre-trained HuggingFace embedding model
print("[INFO] Initializing HuggingFace embedding model...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up the Chroma vector store to persist and store document embeddings
print("[INFO] Setting up Chroma vector store...")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Set up the text splitter to divide documents into manageable chunks
print("[INFO] Setting up text splitter...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=True,
)

# Pydantic model for structuring the search results returned to the user
class SearchResult(BaseModel):
    id: str
    content: str
    metadata: dict

@app.get("/health")
async def health_check():
    """Health check endpoint to ensure the service is running smoothly."""
    print("[INFO] Health check endpoint called.")
    return {"status": "OK"}

@app.post("/documents/process")
async def process_document(file: UploadFile = File(...)):
    """Endpoint to handle the uploading, processing, and storing of PDF documents."""
    print("[INFO] Processing uploaded document...")
    try:
        # Save the uploaded PDF document to a local file
        file_path = os.path.join(DATA_PATH, file.filename)
        print(f"[INFO] Saving uploaded file to: {file_path}")
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load the content of the uploaded PDF file
        print("[INFO] Loading content from the uploaded PDF file...")
        loader = PyPDFLoader(file_path)
        raw_documents = loader.load()

        # Break the document into smaller chunks of text
        print("[INFO] Splitting document into chunks...")
        chunks = text_splitter.split_documents(raw_documents)

        # Generate unique IDs for each chunk of text
        print(f"[INFO] Generating unique IDs for {len(chunks)} chunks...")
        uuids = [str(uuid4()) for _ in range(len(chunks))]

        # Add the document chunks to the vector store for future retrieval
        print("[INFO] Adding chunks to the vector store...")
        vector_store.add_documents(documents=chunks, ids=uuids)

        return {"status": "success", "message": f"Processed {len(chunks)} chunks from the document."}

    except Exception as e:
        print(f"[ERROR] Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {e}")

@app.get("/search", response_model=List[SearchResult])
async def search(query: str = Query(...), k: int = Query(5)):
    """Endpoint to perform semantic search on the stored documents."""
    print(f"[INFO] Performing semantic search for query: '{query}' with top {k} results...")
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        print("[INFO] Preparing search results...")
        results = [
            SearchResult(id=doc.metadata.get("id", "N/A"), content=doc.page_content, metadata=doc.metadata)
            for doc in docs
        ]

        return results

    except Exception as e:
        print(f"[ERROR] Error during search: {e}")
        raise HTTPException(status_code=500, detail=f"Error during search: {e}")
