import streamlit as st

# 1. Core Page Configuration: Must precede all local module imports.
st.set_page_config(
    page_title="Policy & Compliance Assistant", 
    page_icon="⚖️", 
    layout="centered"
)

import logging
# Subsequent local imports deferred until after page configuration.
from utils.retrieval import generate_response

# Configure systematic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """
    Initializes the temporal memory vectors for the Streamlit execution environment.
    Prevents the deletion of conversational history during UI re-renders.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "System initialized. State your query regarding internal policies or compliance."}
        ]
    if "user_id" not in st.session_state:
        st.session_state.user_id = "employee_001" 

def main():
    """
    Primary execution function for the Streamlit interface.
    Handles layout rendering, parameter configuration, and asynchronous data flow.
    """
    st.title("Policy & Compliance Assistant")
    st.markdown("### Enterprise Knowledge Retrieval Agent")
    
    # Sidebar Configuration (System Parameters)
    st.sidebar.header("System Parameters")
    response_mode = st.sidebar.radio(
        "Select Response Modality:",
        ("Concise", "Detailed"),
        index=0,
        help="Determines the lexical density and verbosity of the generated output."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Agent Capabilities:**\n"
        "- Local Vector Retrieval (FAISS)\n"
        "- Long-Document Parsing (PDR)\n"
        "- Web Search Fallback\n"
        "- Persistent Preference Memory"
    )
    
    # State Initialization
    initialize_session_state()
    
    # Render Historical Interaction State
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Primary Input Vector & Execution Flow
    if user_query := st.chat_input("Enter compliance or policy query..."):
        
        # Append and render user query
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Execute Retrieval and Generation Pipeline
        with st.chat_message("assistant"):
            with st.spinner("Executing retrieval and generating response..."):
                try:
                    response_content = generate_response(
                        query=user_query, 
                        mode=response_mode, 
                        user_id=st.session_state.user_id
                    )
                    st.markdown(response_content)
                    
                    # Append system response to temporal memory
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                    
                except Exception as e:
                    error_msg = f"Critical System Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)

if __name__ == "__main__":
    main()