import logging
from mem0 import Memory
from config.config import GOOGLE_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_client():
    """
    Initializes the Mem0 memory client.
    Explicitly defines a 768-dimension vector collection to align with Gemini constraints.
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
            "provider": "gemini",
            "config": {
                "model": "gemini-2.5-flash-lite",
                "api_key": GOOGLE_API_KEY
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

def add_to_memory(user_id: str, user_input: str):
    memory = get_memory_client()
    try:
        memory.add(user_input, user_id=user_id)
        logger.info(f"✅ Memory processed for user: {user_id}")
    except Exception as e:
        logger.error(f"❌ Error adding to memory: {e}")

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