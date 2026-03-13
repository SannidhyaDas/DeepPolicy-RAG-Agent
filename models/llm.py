# import logging
# from langchain_google_genai import ChatGoogleGenerativeAI
# from config.config import GOOGLE_API_KEY

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def get_llm(temperature: float = 0.3):
#     """
#     Initializes and returns the main Gemini LLM.
#     We keep the temperature low (0.3) for RAG to prevent hallucination.
#     """
#     try:
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-2.5-flash-lite", 
#             temperature=temperature,
#             google_api_key=GOOGLE_API_KEY
#         )
#         logger.info(f"✅ LLM (gemini-2.5-flash-lite) initialized successfully with temp={temperature}.")
#         return llm
#     except Exception as e:
#         logger.error(f"❌ CRITICAL ERROR initializing LLM: {e}")
#         raise e
    

import logging
from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm(temperature: float = 0.0):
    """
    Initializes and returns the main Groq LLM (Llama 3.3 70B).
    Provides enterprise-grade scalability and ultra-low latency inference.
    """
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=temperature,
            api_key=GROQ_API_KEY
        )
        logger.info(f"✅ LLM (llama-3.3-70b-versatile) initialized successfully via Groq with temp={temperature}.")
        return llm
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR initializing LLM: {e}")
        raise e