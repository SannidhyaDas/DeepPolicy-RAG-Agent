import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import GOOGLE_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm(temperature: float = 0.3):
    """
    Initializes and returns the main Gemini LLM.
    We keep the temperature low (0.3) for RAG to prevent hallucination.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY
        )
        logger.info(f"✅ LLM (gemini-2.5-flash) initialized successfully with temp={temperature}.")
        return llm
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR initializing LLM: {e}")
        raise e