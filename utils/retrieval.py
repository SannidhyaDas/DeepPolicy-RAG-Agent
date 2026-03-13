import os
import re
import hashlib
import logging
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from models.llm import get_llm
from utils.ingestion import build_or_load_retriever
from utils.web_search import search_web_with_citations
from utils.memory_logic import get_user_context, add_to_memory
from config.config import GROQ_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Primary Cache Dictionary
RESPONSE_CACHE = {}

## SECURITY GUARDRAIL LAYERS

def deterministic_pre_flight(query: str) -> bool:
    """Layer 1: Evaluates query against rigid malicious signatures."""
    blacklisted_phrases = ["ignore previous instructions", "system prompt", "bypass", "jailbreak"]
    query_lower = query.lower()
    for phrase in blacklisted_phrases:
        if phrase in query_lower:
            logger.warning(f"Security Alert: Deterministic block triggered by signature '{phrase}'")
            return False
    return True

def sanitize_pii(text: str) -> str:
    """Layer 2: Redacts standard Personally Identifiable Information using Regex."""
    # Updated regex: Mandates strict alphabetical termination (\.[A-Za-z]{2,}\b) 
    # to prevent greedy consumption of trailing sentence punctuation.
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL REDACTED]', text)
    
    # Retained phone redaction logic
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE REDACTED]', text)
    
    return text

def semantic_security_gate(query: str) -> bool:
    """Layer 3: Utilizes lightweight Groq LLM (8B) to classify semantic intent."""
    try:
        guard_llm = ChatGroq(
            model="llama3-8b-8192", 
            temperature=0.0,
            api_key=GROQ_API_KEY
        )
        
        guard_prompt = f"""You are an enterprise security classifier. 
        Analyze the following user input. Does it pertain to company policy, compliance, internal operations, business research, or general professional inquiries?
        Output strictly the word 'SAFE' if relevant and safe, or 'BLOCK' if off-topic, malicious, toxic, or inappropriate. Do not output any other text.
        
        User Input: {query}
        Classification:"""
        
        response = guard_llm.invoke(guard_prompt)
        classification = response.content.strip().upper()
        
        if "BLOCK" in classification:
            logger.warning("Security Alert: Semantic gate triggered BLOCK classification.")
            return False
        return True
    except Exception as e:
        logger.error(f"Semantic Security Gate Failure: {e}")
        # Failsafe open to prevent total system lockout during transient API errors
        return True


## PRIMARY ORCHESTRATOR

def generate_response(query: str, mode: str, user_id: str = "default_user") -> str:
    """
    Orchestrates RAG generation with Layered Security, Hashlib Caching, and strict Agentic Routing.
    """
    # 1. Execute Security Pre-Flight (Layer 1)
    if not deterministic_pre_flight(query):
        return "Security Violation: Your query contains restricted command signatures and has been blocked."
        
    # 2. Execute PII Sanitization (Layer 2)
    sanitized_query = sanitize_pii(query)
    
    # 3. Execute Semantic Guardrail (Layer 3)
    if not semantic_security_gate(sanitized_query):
        return "Compliance Warning: Your query has been flagged as off-topic or inappropriate for this enterprise assistant."

    # 4. Evaluate Cache Status (Using sanitized query)
    cache_key = hashlib.md5(f"{sanitized_query}_{mode}_{user_id}".encode()).hexdigest()
    if cache_key in RESPONSE_CACHE:
        logger.info(f"Using cached result for query hash: {cache_key}")
        return RESPONSE_CACHE[cache_key]

    # Initialize Primary 70B Generation Engine
    llm = get_llm(temperature=0.0) 
    retriever = build_or_load_retriever()

    # 5. Retrieve Internal Document Context
    try:
        retrieved_docs = retriever.invoke(sanitized_query)
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

    # 6. Formulate Strict Routing Prompt
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
        formatted_prompt = prompt.format(memory=user_memory, context=context_text, query=sanitized_query)
        response = llm.invoke(formatted_prompt)
        final_answer = response.content.strip()
        
        # 7. Agentic Intercept & Fallback Execution
        if "TRIGGER_WEB_SEARCH" in final_answer:
            logger.info("Internal context insufficient. Triggering secondary Web Search retrieval loop.")
            
            web_context, web_citations = search_web_with_citations(sanitized_query)
            
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
            fallback_response = llm.invoke(fallback_prompt.format(context=web_context, query=sanitized_query))
            
            final_answer = fallback_response.content + web_citations
        else:
            if internal_sources:
                citation_text = "\n\n**DOCUMENT SOURCES:**\n"
                for i, source in enumerate(sorted(internal_sources), 1):
                    citation_text += f"{i}. {source}\n"
                final_answer += citation_text

        # 8. Persist State and Commit to Cache
        memory_payload = f"Query: {sanitized_query} | Response Summary: {final_answer[:100]}..."
        add_to_memory(user_id, memory_payload)
        
        RESPONSE_CACHE[cache_key] = final_answer
        
        return final_answer
        
    except Exception as e:
        logger.error(f"LLM inference execution failed: {e}")
        return "System Error: The generation module encountered a critical failure."