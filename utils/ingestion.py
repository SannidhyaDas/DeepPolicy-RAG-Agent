import os
import pickle
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

from models.embeddings import get_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for storage paths
DATA_DIR = "data"
INDEX_DIR = "store/faiss_index"
DOCSTORE_FILE = "store/docstore.pkl"

def load_documents_from_directory(directory_path: str):
    """Loads all PDFs from the specified directory using PyMuPDF (fast!)."""
    docs = []
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.warning(f"Created {directory_path} folder. Please add PDFs.")
        return docs

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            try:
                loader = PyMuPDFLoader(file_path)
                docs.extend(loader.load())
                logger.info(f"Loaded: {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    return docs

def build_or_load_retriever():
    """
    Core PDR Logic: 
    Checks if we already have a saved FAISS index and DocStore.
    If yes: Loads them into memory (fast).
    If no: Processes the PDFs in data/, builds the stores, and saves them.
    """
    embeddings = get_embedding_model()
    
    # Define our splitters
    # Parent: Large chunks to give the LLM full context
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    # Child: Small chunks for highly accurate FAISS vector search
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # 1. CHECK IF ALREADY PROCESSED
    if os.path.exists(INDEX_DIR) and os.path.exists(DOCSTORE_FILE):
        logger.info("Loading existing FAISS index and DocStore from disk...")
        try:
            # allow_dangerous_deserialization is required for FAISS local loading in modern LangChain
            vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            
            with open(DOCSTORE_FILE, "rb") as f:
                store_dict = pickle.load(f)
            
            docstore = InMemoryStore()
            docstore.store = store_dict
            
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=docstore,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
            return retriever
        except Exception as e:
            logger.error(f"Failed to load existing index. Rebuilding... Error: {e}")

    # 2. IF NOT PROCESSED, BUILD FROM SCRATCH
    logger.info("No existing index found. Building new FAISS index and DocStore...")
    docs = load_documents_from_directory(DATA_DIR)
    
    if not docs:
        raise ValueError("No PDFs found in the 'data' directory. Please add some files!")

    # Initialize empty vectorstore and docstore
    vectorstore = FAISS.from_texts(["initialization text"], embeddings) 
    # (We initialize with dummy text because LangChain FAISS requires initial data, 
    # we'll clear it via PDR instantly)
    
    docstore = InMemoryStore()
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # This magic command splits the docs, embeds the children, and stores the parents!
    retriever.add_documents(docs)

    # 3. SAVE TO DISK FOR STREAMLIT TO USE LATER
    if not os.path.exists("store"):
        os.makedirs("store")
        
    vectorstore.save_local(INDEX_DIR)
    with open(DOCSTORE_FILE, "wb") as f:
        pickle.dump(docstore.store, f)
        
    logger.info("✅ FAISS index and DocStore successfully built and saved locally.")
    
    return retriever
