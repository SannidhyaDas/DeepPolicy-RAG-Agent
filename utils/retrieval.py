# import os
# import hashlib
# import logging
# from langchain_core.prompts import PromptTemplate

# from models.llm import get_llm
# from utils.ingestion import build_or_load_retriever
# from utils.web_search import search_web_with_citations
# from utils.memory_logic import get_user_context, add_to_memory

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Primary Cache Dictionary (per user research parameters)
# RESPONSE_CACHE = {}

# def generate_response(query: str, mode: str, user_id: str = "default_user") -> str:
#     """
#     Orchestrates RAG generation with Hashlib Caching, strict Agentic Routing, and explicit citations.
#     """
#     # 1. Evaluate Cache Status
#     cache_key = hashlib.md5(f"{query}_{mode}_{user_id}".encode()).hexdigest()
#     if cache_key in RESPONSE_CACHE:
#         logger.info(f"Using cached result for query hash: {cache_key}")
#         return RESPONSE_CACHE[cache_key]

#     # Initialize LLM with zero temperature for maximum determinism
#     llm = get_llm(temperature=0.0) 
#     retriever = build_or_load_retriever()

#     # 2. Retrieve Internal Document Context
#     try:
#         retrieved_docs = retriever.invoke(query)
#     except Exception as e:
#         logger.error(f"Vector store retrieval failure: {e}")
#         retrieved_docs = []

#     if retrieved_docs:
#         context_text = "\n\n---\n\n".join(
#             [f"Source Document: {os.path.basename(doc.metadata.get('source', 'Local Data'))}\nContent: {doc.page_content}" 
#              for doc in retrieved_docs]
#         )
#     else:
#         context_text = "No internal documents available."

#     user_memory = get_user_context(user_id)

#     if mode == "Concise":
#         style_instruction = "Provide a direct, one-sentence summation. Utilize bullet points strictly when enumerating discrete items."
#     else:
#         style_instruction = "Provide a comprehensive, highly detailed exposition incorporating background context."

#     # 3. Formulate Strict Routing Prompt
#     prompt_template = f"""You are an advanced internal Assistant.
    
#     System Directives:
#     {style_instruction}
#     1. Evaluate the Information Context. If the context does NOT explicitly contain the answer to the User Query, you MUST output ONLY the exact phrase: "TRIGGER_WEB_SEARCH". Do not guess or attempt to provide partial answers.
#     2. If the context DOES contain the answer, formulate your response utilizing ONLY the provided context boundaries.
#     3. MANDATORY: If utilizing internal documents, append your source at the end formatted exactly as: "SOURCES:\n1. [Document Name]".
    
#     Extracted User Memory Context:
#     {{memory}}
    
#     Information Context:
#     {{context}}
    
#     User Query:
#     {{query}}
    
#     Final Answer:"""

#     prompt = PromptTemplate(template=prompt_template, input_variables=["memory", "context", "query"])

#     try:
#         # First Generation Cycle (Internal PDF Evaluation)
#         formatted_prompt = prompt.format(memory=user_memory, context=context_text, query=query)
#         response = llm.invoke(formatted_prompt)
#         final_answer = response.content.strip()
        
#         # 4. Agentic Intercept & Fallback Execution
#         if "TRIGGER_WEB_SEARCH" in final_answer:
#             logger.info("Internal context insufficient. Triggering secondary Web Search retrieval loop.")
            
#             web_context, web_citations = search_web_with_citations(query)
            
#             fallback_prompt_template = f"""You are an advanced Assistant.
#             System Directives:
#             {style_instruction}
#             Answer the user query based ONLY on the Web Search Results Context provided. Do not append manual citations; they will be handled programmatically.
            
#             Web Search Results Context:
#             {{context}}
            
#             User Query:
#             {{query}}
            
#             Final Answer:"""
            
#             fallback_prompt = PromptTemplate(template=fallback_prompt_template, input_variables=["context", "query"])
#             fallback_response = llm.invoke(fallback_prompt.format(context=web_context, query=query))
            
#             # Programmatically append the formatted web citations
#             final_answer = fallback_response.content + web_citations

#         # 5. Persist State and Commit to Cache
#         memory_payload = f"Query: {query} | Response Summary: {final_answer[:100]}..."
#         add_to_memory(user_id, memory_payload)
        
#         RESPONSE_CACHE[cache_key] = final_answer
        
#         return final_answer
        
#     except Exception as e:
#         logger.error(f"LLM inference execution failed: {e}")
#         return "System Error: The generation module encountered a critical failure."

import os
import hashlib
import logging
from langchain_core.prompts import PromptTemplate

from models.llm import get_llm
from utils.ingestion import build_or_load_retriever
from utils.web_search import search_web_with_citations
from utils.memory_logic import get_user_context, add_to_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Primary Cache Dictionary
RESPONSE_CACHE = {}

def generate_response(query: str, mode: str, user_id: str = "default_user") -> str:
    """
    Orchestrates RAG generation with Hashlib Caching, strict Agentic Routing, and programmatic citations.
    """
    # 1. Evaluate Cache Status
    cache_key = hashlib.md5(f"{query}_{mode}_{user_id}".encode()).hexdigest()
    if cache_key in RESPONSE_CACHE:
        logger.info(f"Using cached result for query hash: {cache_key}")
        return RESPONSE_CACHE[cache_key]

    llm = get_llm(temperature=0.0) 
    retriever = build_or_load_retriever()

    # 2. Retrieve Internal Document Context
    try:
        retrieved_docs = retriever.invoke(query)
    except Exception as e:
        logger.error(f"Vector store retrieval failure: {e}")
        retrieved_docs = []

    # Track internal document sources programmatically
    internal_sources = set()
    if retrieved_docs:
        context_blocks = []
        for doc in retrieved_docs:
            source_name = os.path.basename(doc.metadata.get('source', 'Local Document'))
            internal_sources.add(source_name)
            context_blocks.append(f"Source: {source_name}\nContent: {doc.page_content}")
        context_text = "\n\n---\n\n".join(context_blocks)
    else:
        context_text = "No internal documents available."

    user_memory = get_user_context(user_id)

    if mode == "Concise":
        style_instruction = "Provide a direct, one-sentence summation. Utilize bullet points strictly when enumerating discrete items."
    else:
        style_instruction = "Provide a comprehensive, highly detailed exposition incorporating background context."

    # 3. Formulate Strict Routing Prompt
    prompt_template = f"""You are an advanced internal Assistant.
    
    System Directives:
    {style_instruction}
    1. Evaluate the Information Context. If the context does NOT explicitly contain the answer to the User Query, you MUST output ONLY the exact phrase: "TRIGGER_WEB_SEARCH". Do not guess or attempt to provide partial answers.
    2. If the context DOES contain the answer, formulate your response utilizing ONLY the provided context boundaries.
    3. Do not include any manual citations in your response text. Citations are handled programmatically.
    
    Extracted User Memory Context:
    {{memory}}
    
    Information Context:
    {{context}}
    
    User Query:
    {{query}}
    
    Final Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["memory", "context", "query"])

    try:
        # First Generation Cycle (Internal PDF Evaluation)
        formatted_prompt = prompt.format(memory=user_memory, context=context_text, query=query)
        response = llm.invoke(formatted_prompt)
        final_answer = response.content.strip()
        
        # 4. Agentic Intercept & Fallback Execution
        if "TRIGGER_WEB_SEARCH" in final_answer:
            logger.info("Internal context insufficient. Triggering secondary Web Search retrieval loop.")
            
            web_context, web_citations = search_web_with_citations(query)
            
            fallback_prompt_template = f"""You are an advanced Assistant.
            System Directives:
            {style_instruction}
            Answer the user query based ONLY on the Web Search Results Context provided. Do not append manual citations.
            
            Web Search Results Context:
            {{context}}
            
            User Query:
            {{query}}
            
            Final Answer:"""
            
            fallback_prompt = PromptTemplate(template=fallback_prompt_template, input_variables=["context", "query"])
            fallback_response = llm.invoke(fallback_prompt.format(context=web_context, query=query))
            
            # Programmatically append the formatted web citations
            final_answer = fallback_response.content + web_citations
        else:
            # Programmatically append internal document citations
            if internal_sources:
                citation_text = "\n\n**DOCUMENT SOURCES:**\n"
                for i, source in enumerate(sorted(internal_sources), 1):
                    citation_text += f"{i}. {source}\n"
                final_answer += citation_text

        # 5. Persist State and Commit to Cache
        memory_payload = f"Query: {query} | Response Summary: {final_answer[:100]}..."
        add_to_memory(user_id, memory_payload)
        
        RESPONSE_CACHE[cache_key] = final_answer
        
        return final_answer
        
    except Exception as e:
        logger.error(f"LLM inference execution failed: {e}")
        return "System Error: The generation module encountered a critical failure."