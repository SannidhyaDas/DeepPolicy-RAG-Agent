import logging
from mem0 import Memory
from config.config import GOOGLE_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_client():
    """
    Initializes the Mem0 memory client.
    Configured to use Gemini as the extractor brain to save you from needing an OpenAI key.
    """
    config = {
        "llm": {
            "provider": "gemini",
            "config": {
                "model": "gemini-2.5-flash",
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
    Passes the user's message to Mem0. Mem0 will automatically 
    analyze it and decide if there is a fact worth remembering.
    """
    memory = get_memory_client()
    try:
        # We wrap this in a try-except so the main chat doesn't break if memory extraction fails
        memory.add(user_input, user_id=user_id)
        logger.info(f"✅ Memory processed for user: {user_id}")
    except Exception as e:
        logger.error(f"❌ Error adding to memory: {e}")

def get_user_context(user_id: str) -> str:
    """
    Retrieves all remembered facts for a specific user to inject into the final prompt.
    """
    memory = get_memory_client()
    try:
        memories = memory.get_all(user_id=user_id)
        
        if not memories:
            return "No specific user preferences found."
        
        # Format the extracted facts into a clean string
        context = "User Preferences and History:\n"
        for m in memories:
            # Mem0 stores the extracted fact in a 'memory' key
            mem_text = m.get('memory', '')
            if mem_text:
                context += f"- {mem_text}\n"
                
        return context
    except Exception as e:
        logger.error(f"❌ Error retrieving memory: {e}")
        return "Could not retrieve user preferences due to an error."