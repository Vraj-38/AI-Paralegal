import streamlit as st
import os
from rag_chatbot import RAGChatbot
import tempfile

# Set page configuration
st.set_page_config(
    page_title="AI Paralegal",
    page_icon="⚖️",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #2b313e;
        color: white;
    }
    .bot-message {
        background-color: #0e1117;
        color: white;
        border: 1px solid #2b313e;
    }
    .source-info {
        font-size: 0.8rem;
        color: #8c8f94;
        margin-top: 0.5rem;
        border-top: 1px solid #2b313e;
        padding-top: 0.5rem;
    }
    .chat-message div {
        margin-bottom: 0.5rem;
    }
    .user-icon, .bot-icon {
        font-weight: bold;
        color: #00ff95;
    }
    .welcome-text {
        font-size: 1.2rem;
        margin-bottom: 1rem;
        color: #8c8f94;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = RAGChatbot()

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory and return their paths."""
    temp_dir = tempfile.mkdtemp()
    paths = []
    
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        paths.append(temp_path)
    
    return temp_dir, paths

def main():
    st.title("⚖️ AI Paralegal")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📁 Upload Legal Documents")
        uploaded_files = st.file_uploader(
            "Upload your legal documents (PDF)",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing legal documents..."):
                    # Save uploaded files and get their paths
                    temp_dir, pdf_paths = save_uploaded_files(uploaded_files)
                    
                    try:
                        # Load documents into chatbot
                        st.session_state.chatbot.load_from_directory(temp_dir)
                        st.sidebar.success(f"Successfully processed {len(uploaded_files)} documents!")
                    except Exception as e:
                        st.sidebar.error(f"Error processing documents: {str(e)}")
        
        # Add sidebar information
        with st.expander("ℹ️ About AI Paralegal"):
            st.write("""
            I am an AI Paralegal assistant specialized in analyzing legal documents and precedents. I can:
            
            1. Answer general legal questions
            2. Analyze legal documents
            3. Extract relevant information from case files
            4. Find specific details in legal precedents
            
            To get started:
            1. Upload your legal documents (PDFs)
            2. Click 'Process Documents'
            3. Ask me any questions about the documents
            
            You can also chat with me about general topics!
            """)

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div><span class="user-icon">👤 User:</span> {message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            source_section = ""
            if message["sources"]:
                source_section = f"""<div class="source-info">📚 Sources: {message["sources"]}</div>"""
            
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <div><span class="bot-icon">⚖️ AI Paralegal:</span> {message["content"]}</div>
                    {source_section}
                </div>
            """, unsafe_allow_html=True)

    # Chat input
    user_question = st.chat_input("Ask me anything about your legal documents or general questions...")
    
    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        try:
            # Get response from chatbot
            with st.spinner("Analyzing..."):
                response = st.session_state.chatbot.chat(user_question)
            
            # Format sources if they exist
            sources_text = ""
            if response["sources"]:
                sources_text = " | ".join([
                    f"{source['file']} (Page {source['page']})"
                    for source in response["sources"]
                ])
            
            # Add bot response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": sources_text
            })
            
            # Rerun to update chat display
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Display welcome message for first-time users
    if not st.session_state.messages:
        st.markdown("""
            <div class="welcome-text">
                👋 Hello! I'm your AI Paralegal assistant. I can help you with:
                <ul>
                    <li>Analyzing legal documents</li>
                    <li>Finding relevant precedents</li>
                    <li>Answering legal questions</li>
                    <li>General conversation</li>
                </ul>
                Feel free to start a conversation or upload your documents!
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 