import logging
from mem0 import Memory
from config.config import GOOGLE_API_KEY, GROQ_API_KEY
from tenacity import retry, wait_exponential, stop_after_attempt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_client():
    """
    Initializes the Mem0 memory client utilizing a Hybrid Architecture.
    LLM extraction is offloaded to Groq, while Embeddings remain on Gemini.
    """
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "gemini_memory_768",
                "embedding_model_dims": 768
            }
        },
        "llm": {
            "provider": "groq",
            "config": {
                "model": "llama-3.3-70b-versatile",
                "api_key": GROQ_API_KEY
            }
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": "models/gemini-embedding-001",
                "api_key": GOOGLE_API_KEY
            }
        }
    }
    
    try:
        memory = Memory.from_config(config_dict=config)
        return memory
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR initializing Mem0: {e}")
        raise e

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def robust_memory_add(memory_client, user_input, user_id):
    """Executes memory ingestion with automated retries for transient API limits."""
    memory_client.add(user_input, user_id=user_id)

def add_to_memory(user_id: str, user_input: str):
    memory = get_memory_client()
    try:
        robust_memory_add(memory, user_input, user_id)
        logger.info(f"✅ Memory processed for user: {user_id}")
    except Exception as e:
        logger.error(f"❌ Error adding to memory after retries exhausted: {e}")

def get_user_context(user_id: str) -> str:
    memory = get_memory_client()
    try:
        memories = memory.get_all(user_id=user_id)
        
        if not memories:
            return "No specific user preferences found."
        
        context = "User Preferences and History:\n"
        for m in memories:
            if isinstance(m, dict):
                mem_text = m.get('memory', '')
            elif hasattr(m, 'memory'):
                mem_text = getattr(m, 'memory', '')
            else:
                mem_text = str(m)
                
            if mem_text:
                context += f"- {mem_text}\n"
                
        return context
    except Exception as e:
        logger.error(f"❌ Error retrieving memory: {e}")
        return "Could not retrieve user preferences due to an error."