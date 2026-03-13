import logging
from langchain.prompts import PromptTemplate

from models.llm import get_llm
from utils.ingestion import build_or_load_retriever
from utils.web_search import search_web
from utils.memory_logic import get_user_context, add_to_memory

# Configure academic/systematic logging for tracking execution states
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_response(query: str, mode: str, user_id: str = "default_user") -> str:
    """
    Orchestrates the retrieval and generation pipeline.
    
    Workflow:
    1. Instantiates the LLM and the Document Retriever.
    2. Queries the local vector store (FAISS) using the Parent Document Retriever.
    3. Evaluates retrieval yield; if insufficient, triggers the DuckDuckGo web search fallback.
    4. Retrieves historical user context via Mem0.
    5. Formulates a deterministic prompt incorporating context, memory, and the requested verbosity mode.
    6. Executes the LLM inference and records the interaction payload to memory.
    """
    
    # 1. Initialize Core Components
    llm = get_llm(temperature=0.2)  # Low temperature to minimize hallucination variance
    retriever = build_or_load_retriever()

    # 2. Execute Internal Knowledge Retrieval
    try:
        # The invoke method replaces get_relevant_documents in modern LangChain
        retrieved_docs = retriever.invoke(query)
        logger.info(f"Successfully retrieved {len(retrieved_docs)} parent documents from local FAISS index.")
    except Exception as e:
        logger.error(f"Vector store retrieval failure: {e}")
        retrieved_docs = []

    # 3. Evaluate and Apply Web Search Fallback Condition
    context_source = "Internal Documents"
    if not retrieved_docs:
        logger.info("Internal document yield is zero. Initiating DuckDuckGo fallback protocol.")
        web_results = search_web(query)
        context_text = web_results
        context_source = "Web Search"
    else:
        # Aggregate retrieved parent document contents
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    # 4. Fetch User Context Memory
    user_memory = get_user_context(user_id)

    # 5. Determine Lexical Output Parameters
    if mode == "Concise":
        style_instruction = "Provide a direct, one-sentence academic summation. Utilize bullet points strictly when enumerating discrete items."
    else:
        style_instruction = "Provide a comprehensive, highly detailed academic exposition incorporating background context and analytical explanations."

    # 6. Construct the Deterministic Prompt Template
    prompt_template = f"""You are an advanced internal Policy & Compliance Assistant.
    
    System Directives:
    {style_instruction}
    Formulate your response strictly utilizing the provided context boundaries. If the target information is absent from the context, state "Information not available in current context" and refrain from predictive extrapolation.
    
    Extracted User Memory Context:
    {{memory}}
    
    Information Context ({context_source}):
    {{context}}
    
    User Query:
    {{query}}
    
    Final Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["memory", "context", "query"]
    )

    # 7. Execute LLM Inference
    try:
        formatted_prompt = prompt.format(memory=user_memory, context=context_text, query=query)
        # In modern LangChain, invoke is preferred over __call__ or predict
        response = llm.invoke(formatted_prompt)
        final_answer = response.content
        
        # 8. Append Interaction to Persistent Memory State
        memory_payload = f"Query: {query} | Response Summary: {final_answer[:100]}..."
        add_to_memory(user_id, memory_payload)
        
        return final_answer
        
    except Exception as e:
        logger.error(f"LLM inference execution failed: {e}")
        return "System Error: The generation module encountered a critical failure. Please review the application logs."