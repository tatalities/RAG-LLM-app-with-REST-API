from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from dotenv import load_dotenv

# Load environment variables
print("[INFO] Loading environment variables...")
load_dotenv()

# Configuration paths
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Initialize embeddings and Chroma vector store
print("[INFO] Initializing embeddings and Chroma vector store...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Load PDF documents
print(f"[INFO] Loading PDF documents from {DATA_PATH}...")
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()
print(f"[INFO] Loaded {len(raw_documents)} documents.")

# Split documents into chunks
print("[INFO] Splitting documents into smaller chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=True,
)
chunks = text_splitter.split_documents(raw_documents)
print(f"[INFO] Split documents into {len(chunks)} chunks.")

# Generate unique IDs for each chunk
print("[INFO] Generating unique IDs for each chunk...")
uuids = [str(uuid4()) for _ in range(len(chunks))]

# Add chunks to vector store in batches
print("[INFO] Adding chunks to the vector store in batches...")
batch_size = 166
for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i:i + batch_size]
    batch_uuids = uuids[i:i + batch_size]
    vector_store.add_documents(documents=batch_chunks, ids=batch_uuids)
    print(f"[INFO] Added batch {i // batch_size + 1} to vector store.")

print("[INFO] All chunks added to vector store successfully.")
