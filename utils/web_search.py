import logging
from duckduckgo_search import DDGS

# Configure systematic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CitationTracker:
    """Class to manage and format web citations precisely as shown in research parameters."""
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
        citations = ["\n\nSOURCES:"]
        for url, info in sorted(self.sources.items(), key=lambda x: x[1]['number']):
            citations.append(f"{info['number']}. [{info['title']}]: {info['url']}")
        return "\n".join(citations)

def search_web_with_citations(query: str, max_results: int = 3):
    """
    Executes web search, applies citation tracking, and returns both context and formatted citations.
    """
    logger.info(f"Initiating web search fallback for query: {query}")
    tracker = CitationTracker()
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            return "No useful results found on the web.", ""
        
        context_blocks = []
        for res in results:
            title = res.get('title', 'No Title')
            body = res.get('body', 'No Description')
            href = res.get('href', 'No URL')
            
            cite_num = tracker.add_source(url=href, title=title)
            context_blocks.append(f"Source [{cite_num}]: {body}")
        
        formatted_context = "Web Search Results Context:\n" + "\n\n".join(context_blocks)
        citations_text = tracker.format_citations()
        
        logger.info("Web search completed successfully.")
        return formatted_context, citations_text
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return "Web search is currently unavailable.", ""