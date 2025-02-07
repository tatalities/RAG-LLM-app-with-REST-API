from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import os

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
print("[INFO] Loading environment variables...")
load_dotenv()

# Configuration
DATA_PATH = r"data"
CHROMA_PATH = "./chroma_db"
# Initialize embedding model
print("[INFO] Initializing HuggingFace embedding model...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize the LLM (large language model) for responses
print("[INFO] Initializing ChatOpenAI model...")
llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

print("[INFO] Setting up Chroma vector store for document retrieval...")
try:
    # Initialize Chroma vector store
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings_model,
        persist_directory=CHROMA_PATH,
    )

    # Check if the directory exists and contains data
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print("[INFO] Chroma vector store is connected and contains data.")
    else:
        print("[INFO] Chroma vector store is connected but is empty.")
except Exception as e:
    print(f"[ERROR] Failed to connect to Chroma vector store: {e}")

# Configure the retriever to return a fixed number of relevant documents
num_results = 15
print(f"[INFO] Configuring retriever to return {num_results} relevant documents...")
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# Function to handle incoming messages and provide responses
def stream_response(message, history):
    print("[INFO] Retrieving relevant documents for the user's query...")
    # Retrieve relevant documents based on the user's query
    docs = retriever.invoke(message)

    # Combine the content of retrieved documents into 'knowledge'
    print("[INFO] Preparing knowledge base from retrieved documents...")
    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content + "\n\n"

    # Construct prompt for the LLM using the message, history, and knowledge
    if message:
        print("[INFO] Generating response using the LLM...")
        rag_prompt = f"""
        You are a Q&A assistant at a mobile phone shop answering questions
        based only on the provided knowledge. Do not mention anything outside of this.

        The question: {message}
        Previous conversation: {history}
        Knowledge to base the answer on: {knowledge}

        Answer comprehensively, considering all provided information.
        """

        # Stream the response to the Gradio app in real-time
        partial_message = ""
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# Initialize Gradio interface for chat
print("[INFO] Initializing Gradio chat interface...")
chatbot = gr.ChatInterface(
    stream_response,
    textbox=gr.Textbox(placeholder="Send to the LLM...", container=False, autoscroll=True, scale=7),
)

# Launch the Gradio app
print("[INFO] Launching Gradio app...")
chatbot.launch(server_name="0.0.0.0", server_port=7860, share=True)