import logging
from duckduckgo_search import DDGS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_web(query: str, max_results: int = 3) -> str:
    """
    Searches the web using DuckDuckGo.
    Returns a formatted string of the top results.
    Wrapped in try-except to handle network or rate-limit issues gracefully.
    """
    logger.info(f"Initiating web search for: {query}")
    try:
        # DDGS is the DuckDuckGo Search client
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            return "No useful results found on the web."
        
        # Format the results into a readable text block for the LLM
        formatted_results = "Web Search Results:\n"
        for i, res in enumerate(results):
            title = res.get('title', 'No Title')
            body = res.get('body', 'No Description')
            formatted_results += f"{i+1}. {title}: {body}\n"
        
        logger.info("✅ Web search completed successfully.")
        return formatted_results
        
    except Exception as e:
        logger.error(f"❌ Web search failed (Rate limit or network error): {e}")
        return "Web search is currently unavailable due to a network error."