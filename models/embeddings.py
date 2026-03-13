import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.config import GOOGLE_API_KEY

# Setup basic logging to facilitate systematic debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embedding_model():
    """
    Initializes and returns the Google Generative AI Embedding model.
    Utilizes the stable gemini-embedding-001 model to ensure API compliance.
    """
    try:
        # Replaced the deprecated text-embedding-004 with the current standard
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=GOOGLE_API_KEY
        )
        logger.info("Embedding model initialized successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Critical error initializing embedding model: {e}")
        raise e