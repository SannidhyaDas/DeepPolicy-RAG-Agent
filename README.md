# DeepPolicy: Enterprise Policy & Compliance RAG Agent

## I. Executive Summary
DeepPolicy is a production-grade Retrieval-Augmented Generation (RAG) agent designed specifically for enterprise knowledge retrieval, HR policy analysis, and compliance verification. Engineered with a highly scalable hybrid-LLM architecture, it ensures ultra-low latency inference, strict data sanitization, and deterministic citation tracking. 

## II. System Architecture

The application utilizes a decoupled hybrid infrastructure to optimize API quotas, reduce latency, and ensure fault tolerance.

* **Reasoning & Generation Engine:** Offloaded to **Groq** (`llama-3.3-70b-versatile`) for high-speed, complex semantic reasoning and adherence to strict generative formatting.
* **Vectorization Engine:** Retained on **Google Gemini** (`gemini-embedding-001`) to handle high-dimensional mathematical embeddings efficiently without exhausting primary LLM quotas.
* **Information Retrieval:** Local document parsing via **FAISS** (Facebook AI Similarity Search) and `PyMuPDF`, augmented by an agentic fallback to the **Tavily AI** Search API for real-time external compliance data.
* **Persistent State Memory:** Powered by **Mem0** (backed by a local Qdrant database), enabling the agent to retain cross-session user preferences and contextual history.

## III. Tri-Layer Security Guardrails

To meet enterprise compliance standards and prevent quota exhaustion, the system implements a strict, sequential interception pipeline before any computationally expensive vector retrieval occurs.

1. **Layer 1: Deterministic Pre-Flight Hook:** A zero-latency regex evaluation that blocks rigid prompt injection vectors (e.g., "ignore previous instructions", "system prompt").
2. **Layer 2: PII Sanitization Middleware:** Automated structural redaction of Personally Identifiable Information (email addresses and phone numbers) to prevent sensitive data leakage into the vector store or external APIs.
3. **Layer 3: Semantic Security Gate:** A specialized, ultra-fast classification execution using Groq's `llama3-8b-8192` model. This gate deterministically blocks general knowledge queries (e.g., arithmetic, trivia, weather, coding requests) and permits only enterprise-relevant policy inquiries.

## IV. Local Deployment Protocol

### Prerequisites
* Python 3.10 - 3.12 (Strictly `<3.13` for C++ binary compatibility)
* API Keys: Groq, Google Gemini, Tavily

### Installation Initialization
1. Clone the repository and instantiate a virtual environment:
   ```bash
   git clone https://github.com/SannidhyaDas/deeppolicy-rag-agent.git
   cd deeppolicy-rag-agent
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Synchronize dependencies (Note: numpy<2.0.0 is strictly required to prevent FAISS C-API binary incompatibility):
   ```bash
   pip install -r requirements.txt
   ```
3. Construct the environmental payload. Create a .env file in the root directory:
   ```bash
   GOOGLE_API_KEY="AIzaSy..."
   TAVILY_API_KEY="tvly-..."
   GROQ_API_KEY="gsk_..."
   ```
 4. Execute the application:
    ```bash
    streamlit run app.py
    ```
## V. Local Deployment Protocol 

The repository includes an automated testing matrix (pytest) utilizing unittest.mock to validate the boundaries of the security guardrails without consuming live API quotas.

To execute the verification suite:
```bash
pytest tests/test_guardrails.py -v
```
## VI. Local Deployment Protocol 

To successfully deploy this architecture to a managed Linux container, the following infrastructural configurations are strictly required to compile the underlying C++ machine learning frameworks.

1. requirements.txt: Ensure faiss-cpu==1.8.0 and numpy<2.0.0 are declared.

2. packages.txt: Ensure a packages.txt file exists in the repository root containing the explicit OpenMP dependency:
```bash
libomp-dev
```
3. Secrets Management: Input the API keys into the Streamlit Advanced Settings > Secrets interface matching the .env structure.
