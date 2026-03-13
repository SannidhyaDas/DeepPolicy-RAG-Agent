# import os
# import json
# import logging
# from langchain_tavily import TavilySearch
# from config.config import TAVILY_API_KEY

# # Set the environment variable required by LangChain's Tavily tool
# os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# # Configure systematic logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class CitationTracker:
#     """Class to manage and format web citations."""
#     def __init__(self):
#         self.sources = {}
#         self.citation_counter = 1

#     def add_source(self, url: str, title: str = None) -> int:
#         if url not in self.sources:
#             self.sources[url] = {
#                 'number': self.citation_counter,
#                 'title': title or "Web Source",
#                 'url': url
#             }
#             self.citation_counter += 1
#         return self.sources[url]['number']

#     def format_citations(self) -> str:
#         if not self.sources:
#             return ""
#         citations = ["\n\n**WEB SOURCES:**"]
#         for url, info in sorted(self.sources.items(), key=lambda x: x[1]['number']):
#             citations.append(f"{info['number']}. [{info['title']}]({info['url']})")
#         return "\n".join(citations)

# def search_web_with_citations(query: str, max_results: int = 3):
#     """
#     Executes enterprise-grade web search using Tavily AI API.
#     Implements JSON deserialization for modernized LangChain compatibility.
#     """
#     logger.info(f"Initiating Tavily API search fallback for query: {query}")
#     tracker = CitationTracker()
    
#     try:
#         search_tool = TavilySearch(max_results=max_results)
#         results = search_tool.invoke(query)
        
#         if not results:
#             logger.warning("Tavily search returned empty results.")
#             return "No useful results found on the web.", ""
            
#         # Convert JSON string to structured Python list
#         if isinstance(results, str):
#             try:
#                 results = json.loads(results)
#             except json.JSONDecodeError:
#                 logger.error("Failed to parse Tavily response as structured JSON.")
#                 return "Web search results could not be parsed.", ""
                
#         context_blocks = []
#         for res in results:
#             # Type safety check to ensure iterable is a dictionary
#             if not isinstance(res, dict):
#                 continue
                
#             body = res.get('content', 'No Description') 
#             href = res.get('url', 'No URL') 
#             title = href.split('/')[2] if '//' in href else 'Web Context'
            
#             cite_num = tracker.add_source(url=href, title=title)
#             context_blocks.append(f"Source [{cite_num}]: {body}")
        
#         formatted_context = "Web Search Results Context:\n" + "\n\n".join(context_blocks)
#         citations_text = tracker.format_citations()
        
#         logger.info("✅ Tavily Web search completed successfully.")
#         return formatted_context, citations_text
        
#     except Exception as e:
#         logger.error(f"❌ Tavily Web search failed: {e}")
#         return "Web search is currently unavailable.", ""

import os
import json
import logging
from langchain_tavily import TavilySearch
from config.config import TAVILY_API_KEY

# Set the environment variable required by LangChain's Tavily tool
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Configure systematic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CitationTracker:
    """Class to manage and format web citations."""
    def __init__(self):
        self.sources = {}
        self.citation_counter = 1

    def add_source(self, url: str, title: str = None) -> int:
        if url not in self.sources:
            self.sources[url] = {
                'number': self.citation_counter,
                'title': title or "Web Source",
                'url': url
            }
            self.citation_counter += 1
        return self.sources[url]['number']

    def format_citations(self) -> str:
        if not self.sources:
            return ""
        citations = ["\n\n**WEB SOURCES:**"]
        for url, info in sorted(self.sources.items(), key=lambda x: x[1]['number']):
            citations.append(f"{info['number']}. [{info['title']}]({info['url']})")
        return "\n".join(citations)

def search_web_with_citations(query: str, max_results: int = 3):
    """
    Executes enterprise-grade web search using Tavily AI API.
    Implements robust JSON deserialization and structure mapping.
    """
    logger.info(f"Initiating Tavily API search fallback for query: {query}")
    tracker = CitationTracker()
    
    try:
        search_tool = TavilySearch(max_results=max_results)
        results = search_tool.invoke(query)
        
        if not results:
            logger.warning("Tavily search returned empty results.")
            return "No useful results found on the web.", ""
            
        # 1. Deserialize JSON string to Python object
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except json.JSONDecodeError:
                logger.error("Failed to parse Tavily response as structured JSON.")
                return "Web search results could not be parsed.", ""
                
        # 2. Extract nested results list if wrapped in a dictionary API payload
        if isinstance(results, dict) and "results" in results:
            results = results["results"]
        elif isinstance(results, dict):
            # Fallback for unexpected dictionary structures
            results = [results]
            
        if not isinstance(results, list):
            results = []
                
        # 3. Process the extracted list of result dictionaries
        context_blocks = []
        for res in results:
            if not isinstance(res, dict):
                continue
                
            body = res.get('content') or res.get('snippet') or 'No Description'
            href = res.get('url') or res.get('link') or 'No URL'
            
            # Extract domain name for a cleaner title
            title = href.split('/')[2] if '//' in href else 'Web Context'
            
            cite_num = tracker.add_source(url=href, title=title)
            context_blocks.append(f"Source [{cite_num}]: {body}")
        
        formatted_context = "Web Search Results Context:\n" + "\n\n".join(context_blocks)
        citations_text = tracker.format_citations()
        
        logger.info("✅ Tavily Web search completed successfully.")
        return formatted_context, citations_text
        
    except Exception as e:
        logger.error(f"❌ Tavily Web search failed: {e}")
        return "Web search is currently unavailable.", ""