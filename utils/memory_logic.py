import logging
from mem0 import Memory
from config.config import GOOGLE_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_client():
    """
    Initializes the Mem0 memory client.
    Configured to use Gemini for BOTH the LLM and the Embedder 
    to completely avoid OpenAI dependencies.
    """
    config = {
        "llm": {
            "provider": "gemini",
            "config": {
                "model": "gemini-2.5-flash",
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
    """
    Passes the user's message to Mem0.
    """
    memory = get_memory_client()
    try:
        memory.add(user_input, user_id=user_id)
        logger.info(f"✅ Memory processed for user: {user_id}")
    except Exception as e:
        logger.error(f"❌ Error adding to memory: {e}")

# def get_user_context(user_id: str) -> str:
#     """
#     Retrieves all remembered facts for a specific user.
#     """
#     memory = get_memory_client()
#     try:
#         memories = memory.get_all(user_id=user_id)
        
#         if not memories:
#             return "No specific user preferences found."
        
#         context = "User Preferences and History:\n"
#         for m in memories:
#             mem_text = m.get('memory', '')
#             if mem_text:
#                 context += f"- {mem_text}\n"
                
#         return context
#     except Exception as e:
#         logger.error(f"❌ Error retrieving memory: {e}")
#         return "Could not retrieve user preferences due to an error."

def get_user_context(user_id: str) -> str:
    """
    Retrieves all remembered facts for a specific user.
    Bulletproofed to handle different Mem0 version output formats.
    """
    memory = get_memory_client()
    try:
        memories = memory.get_all(user_id=user_id)
        
        if not memories:
            return "No specific user preferences found."
        
        context = "User Preferences and History:\n"
        for m in memories:
            # Safely handle both Dictionary and String formats
            if isinstance(m, dict):
                mem_text = m.get('memory', '')
            elif hasattr(m, 'memory'): # Handle custom object types
                mem_text = getattr(m, 'memory', '')
            else:
                mem_text = str(m)
                
            if mem_text:
                context += f"- {mem_text}\n"
                
        return context
    except Exception as e:
        logger.error(f"❌ Error retrieving memory: {e}")
        return "Could not retrieve user preferences due to an error."