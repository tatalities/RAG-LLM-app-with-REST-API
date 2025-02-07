# Real-Time Q&A Assistant with RAG and LLM

## System Overview & Architecture

This project is an AI-driven real-time Question and Answering (Q&A) assistant that leverages **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)** for natural language understanding and document retrieval. It integrates **HuggingFace embeddings** for semantic understanding and **Chroma** as the vector store for efficient retrieval of relevant document chunks.

### **Architecture Overview:**
- **Frontend**: A Gradio UI that facilitates real-time interactions with the Q&A assistant.
- **Backend**: FastAPI for processing document uploads and handling queries.
- **Vector Store**: Chroma, a vector store used to store document embeddings and perform efficient retrieval during queries.
- **Embeddings**: HuggingFace's `all-MiniLM-L6-v2` embeddings model is used to embed document text and query vectors for semantic search.
- **LLM**: OpenAI's GPT model (via Langchain) for generating answers based on retrieved documents.
- **Document Processing**: PDF documents are loaded and split into manageable chunks using Langchain's `PyPDFLoader` and `RecursiveCharacterTextSplitter`.

---

## Tools & Libraries Used

- **Langchain**: A framework for building LLM-based applications, used for vector store management, document processing, and integration with the LLM.
- **Chroma**: A vector database used to persist document embeddings and provide efficient retrieval during query processing.
- **HuggingFace**: Used for embedding models (`all-MiniLM-L6-v2`) for document and query vectors.
- **FastAPI**: A web framework to build the backend for document processing, health check, and search functionality.
- **Gradio**: A Python library for creating real-time user interfaces for machine learning models, used here to interact with the Q&A assistant.
- **Docker**: Used to containerize and run the application in an isolated environment.
- **Uvicorn**: ASGI server used to run FastAPI for serving APIs.
- **Poetry**: A dependency management tool used to manage Python packages and ensure reproducibility.

---

## Challenges Faced

### 1. **Choosing a Free Vector Embedding Model**
   One of the significant challenges was finding a reliable and free vector embedding model that could provide sufficient performance for document retrieval tasks. After several rounds of experimentation, I settled on the **`all-MiniLM-L6-v2`** model from HuggingFace, which offered a good balance between quality and efficiency, given resource constraints.

### 2. **Finding Optimal Chunk Size and Chunk Overlap**
   Splitting documents into smaller, meaningful chunks without losing context was challenging. The chunk size and overlap were tuned through trial and error to achieve the best results. The optimal parameters were found to be:
   - **Chunk Size**: 250 characters
   - **Chunk Overlap**: 50 characters
   This ensures that the chunks are small enough for efficient retrieval while maintaining enough context for meaningful answers.

### 3. **Selecting the Right `k` for Retrieval**
   Tuning the number of documents to retrieve (`k`) during the search phase was crucial to balance between response quality and speed. Experimentation led to the decision of setting `k = 15`, as this provided relevant results without overloading the response time.

### 4. **Integrating Real-Time RAG with UI**
   One of the interesting challenges was enabling real-time interaction with the LLM and RAG system in a user-friendly way. This required careful integration of the Gradio interface with the Langchain framework to stream results in real time, which posed some unique technical challenges in terms of handling large amounts of text efficiently.

### 5. **Fixing Docker Dependencies**
   Initially, the project used `pip` for dependency management, but this led to issues with version conflicts and reproducibility. To address this, I migrated to **Poetry**, which provides better dependency resolution and lock files for consistent builds. This transition required refactoring the Dockerfile and ensuring all dependencies were correctly installed in the container.

### 6. **Running Files Locally from Docker Container**
   Another challenge was ensuring that files (e.g., uploaded documents) could be accessed and processed correctly when running the application inside a Docker container. This required careful volume mounting and path configuration in Docker to ensure seamless file handling between the host and the container.

---
## Steps to Build, Run, and Test

### 1. **Run with Docker**

You can run the application with Docker by using the following commands:

```bash
docker pull tatalities/rag-llm:1.0
```

```bash
docker run -d -p 7860:7860 -p 8000:8000 tatalities/rag-llm:1.0
```

**After docker run, please wait for around 2-3 minutes before you run localhost.**

To inspect logs while waiting, 
```bash
docker ps
```
Get the container id, and then:
```bash
docker logs <container-id>
```


### This will start the Gradio UI to interact with chatbot, which will be accessible at http://localhost:7860.

### The FastAPI service will be available at http://localhost:8000/docs, where you can check system health, upload documents and perform semantic search queries.



