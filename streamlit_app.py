import streamlit as st
import os
import json
import requests
import re
from rag_chatbot import RAGChatbot
from doc_draft import (
    WritPetitionRequest, AffidavitRequest, PatentApplicationRequest,
    AnnexureRequest, WitnessStatementRequest, ExhibitRequest,
    ForensicReportRequest, ExpertOpinionRequest
)
from embeddings import process_uploaded_file

# Set page configuration
st.set_page_config(
    page_title="AI Paralegal",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Set sidebar to be collapsed by default
)

# Initialize session state variables if they don't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Home"

if "document_response" not in st.session_state:
    st.session_state.document_response = None

if "document_filename" not in st.session_state:
    st.session_state.document_filename = None

if "document_type" not in st.session_state:
    st.session_state.document_type = None

if "upload_status" not in st.session_state:
    st.session_state.upload_status = None

# Initialize RAG chatbot
@st.cache_resource
def get_chatbot():
    return RAGChatbot()

chatbot = get_chatbot()

# Define API base URL
API_BASE_URL = "http://localhost:8050"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E5E7EB;
        border-left: 5px solid #1E3A8A;
        color: #111827;
        font-weight: 500;
    }
    .assistant-message {
        background-color: #1E3A8A;
        border-left: 5px solid #10B981;
        color: white;
    }
    .sender {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .message-content {
        white-space: pre-wrap;
        font-family: 'Times New Roman', Times, serif;
        line-height: 1.5;
        text-align: justify;
    }
    .legal-text p {
        margin-bottom: 0.8em;
        text-indent: 1em;
    }
    .legal-text ol, .legal-text ul {
        margin-left: 1.5em;
        margin-bottom: 0.8em;
    }
    .form-container {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Make the sidebar narrower */
    [data-testid="stSidebar"] {
        width: 200px !important;
    }
    
    /* Feature boxes with better contrast */
    .feature-box {
        padding: 16px;
        border-radius: 10px;
        height: 200px;
        margin-bottom: 10px;
        border: 1px solid #E5E7EB;
    }
    
    .feature-box-1 {
        background-color: #D1D5DB;
        color: #1F2937;
    }
    
    .feature-box-2 {
        background-color: #CBD5E1;
        color: #1F2937;
    }
    
    .feature-box-3 {
        background-color: #E2E8F0;
        color: #1F2937;
    }
    
    .feature-box h4 {
        color: #1E3A8A;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    /* Custom styling for input elements */
    .stTextArea textarea {
        background-color: #F8FAFC;
        border: 2px solid #CBD5E1;
        color: #111827;
        font-size: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #1E3A8A;
        box-shadow: 0 0 0 2px rgba(30, 58, 138, 0.2);
    }
    
    .stTextInput input {
        background-color: #F8FAFC;
        border: 2px solid #CBD5E1;
        color: #111827;
        font-size: 1rem;
    }
    
    .stTextInput input:focus {
        border-color: #1E3A8A;
        box-shadow: 0 0 0 2px rgba(30, 58, 138, 0.2);
    }
    
    /* Style for form submit buttons */
    .stButton button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
    }
    
    .stButton button:hover {
        background-color: #1E40AF;
    }
    
    /* Style for download buttons */
    [data-testid="stDownloadButton"] button {
        background-color: #10B981;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        width: 100%;
        margin-top: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="stDownloadButton"] button:hover {
        background-color: #059669;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'>AI Paralegal</div>", unsafe_allow_html=True)

# Simplified sidebar navigation
with st.sidebar:
    # Simple navigation buttons in sequence
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Home button
    if st.button("🏠 Home", key="home_btn", use_container_width=True,
               type="secondary" if st.session_state.current_tab != "Home" else "primary"):
        st.session_state.current_tab = "Home"
        st.rerun()
    
    # Chat button
    if st.button("💬 Chatbot", key="chat_btn", use_container_width=True,
               type="secondary" if st.session_state.current_tab != "Chat" else "primary"):
        st.session_state.current_tab = "Chat"
        st.rerun()
    
    # Document Generation button
    if st.button("📄 DocGenerator", key="doc_btn", use_container_width=True,
               type="secondary" if st.session_state.current_tab != "Document Generation" else "primary"):
        st.session_state.current_tab = "Document Generation"
        st.rerun()
    
    # Knowledge Base button
    if st.button("📚 Knowledge Base", key="kb_btn", use_container_width=True,
               type="secondary" if st.session_state.current_tab != "Knowledge Base" else "primary"):
        st.session_state.current_tab = "Knowledge Base"
        st.rerun()

# Helper function to clean responses from markdown artifacts
def clean_legal_text(text):
    """Clean text from markdown artifacts to ensure professional legal formatting"""
    # Remove extra newlines (more than 2 consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove asterisks used for bold/italic in markdown
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
    
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bullet points and replace with proper indentation
    text = re.sub(r'^\s*[-*•]\s+', '    • ', text, flags=re.MULTILINE)
    
    return text

# Home page content
if st.session_state.current_tab == "Home":
    st.markdown("<div class='sub-header'>Welcome to AI Paralegal </div>", unsafe_allow_html=True)
    
    # Main welcome content with better contrast
    st.markdown("""
    <div style="background-color: #F3F4F6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #1E3A8A;">Your AI-Powered Legal Assistant</h3>
        <p style="color: #111827; font-weight: 500;">
        AI Paralegal is designed to help legal professionals streamline their workflow by providing
        intelligent document analysis, answering legal questions, and generating legal documents.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features in columns with improved contrast
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box feature-box-1">
            <h4>💬 Legal Document Q&A</h4>
            <p>
            Ask questions about your legal documents and get accurate answers based on document content. 
            Our AI assistant can analyze uploaded documents and extract relevant information.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box feature-box-2">
            <h4>📄 Document Generation</h4>
            <p>
            Generate professional legal documents including Writ Petitions, Affidavits, Patent Applications, 
            and more with our AI-powered document generation tool.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box feature-box-3">
            <h4>📚 Knowledge Base</h4>
            <p>
            Build your own legal knowledge base by uploading PDF documents. 
            Our system processes and indexes these documents, making them searchable and accessible.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # How to get started
    st.markdown("### How to Get Started")
    st.markdown("""
    1. **Upload Documents**: Add PDF files to your knowledge base using the Knowledge Base section
    2. **Ask Questions**: Use the Chatbot to ask questions about your documents
    3. **Generate Documents**: Use the Document Generation section to create legal documents
    """)
    
    # Quick start buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Go to Chatbot", use_container_width=True):
            st.session_state.current_tab = "Chat"
            st.rerun()
    
    with col2:
        if st.button("Generate Documents", use_container_width=True):
            st.session_state.current_tab = "Document Generation"
            st.rerun()
    
    with col3:
        if st.button("Manage Knowledge Base", use_container_width=True):
            st.session_state.current_tab = "Knowledge Base"
            st.rerun()

# Chat functionality
elif st.session_state.current_tab == "Chat":
    st.markdown("<div class='sub-header'>Legal Document Q&A</div>", unsafe_allow_html=True)
    
    # Display chat history with improved formatting
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"""
            <div class='chat-message user-message'>
                <div class='sender'>You:</div>
                <div class='message-content'>{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Clean and format assistant messages for professional legal display
            clean_content = clean_legal_text(content)
            st.markdown(f"""
            <div class='chat-message assistant-message'>
                <div class='sender'>AI Paralegal:</div>
                <div class='message-content legal-text'>{clean_content}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Input for new message
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Type your message:", key="user_input", height=100)
        submit_button = st.form_submit_button("Send")
        
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get response from chatbot
        with st.spinner("AI Paralegal is thinking..."):
            try:
                response = chatbot.chat(user_input)
                
                # Make sure response is properly formatted even if it contains an error
                if isinstance(response, dict) and "answer" in response:
                    answer_text = response["answer"]
                else:
                    answer_text = "Error: Received an invalid response format. Please try again."
                    
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
            except Exception as e:
                error_message = f"Error processing your request: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        
        # Rerun to update the UI
        st.rerun()

# Document Generation functionality
elif st.session_state.current_tab == "Document Generation":
    st.markdown("<div class='sub-header'>Legal Document Generation</div>", unsafe_allow_html=True)
    
    # Document type selection
    doc_types = [
        "Writ Petition", "Affidavit", "Patent Application", "Annexure", 
        "Witness Statement", "Exhibit", "Forensic Report", "Expert Opinion"
    ]
    selected_doc = st.selectbox("Select Document Type:", doc_types)
    
    # Display appropriate form based on selection
    if selected_doc == "Writ Petition":
        with st.form("writ_petition_form"):
            st.markdown("<div class='form-container'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                case_number = st.text_input("Case Number:")
                year = st.number_input("Year:", min_value=1900, max_value=2100, value=2023)
                petitioner = st.text_input("Petitioner:")
                respondent = st.text_input("Respondent:")
            
            with col2:
                court_name = st.text_input("Court Name:")
                jurisdiction = st.text_input("Jurisdiction:")
                legal_grounds = st.text_area("Legal Grounds:")
                relief_sought = st.text_area("Relief Sought:")
            
            benefits_offered = st.text_area("Benefits Offered:")
            supporting_docs = st.text_area("Supporting Documents (one per line):")
            supporting_documents = [doc.strip() for doc in supporting_docs.split("\n") if doc.strip()]
            
            submit = st.form_submit_button("Generate Writ Petition")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if submit:
                # Create request data
                request_data = WritPetitionRequest(
                    case_number=case_number,
                    year=year,
                    petitioner=petitioner,
                    respondent=respondent,
                    court_name=court_name,
                    jurisdiction=jurisdiction,
                    legal_grounds=legal_grounds,
                    relief_sought=relief_sought,
                    benefits_offered=benefits_offered,
                    supporting_documents=supporting_documents
                )
                
                # Send API request for Writ Petition
                with st.spinner("Generating Writ Petition..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/generate/writ_petition",
                            json=request_data.model_dump()
                        )
                        if response.status_code == 200:
                            st.session_state.document_response = response
                            
                            # Get filename from response headers if available
                            filename = "Writ_Petition.docx"
                            if "Content-Disposition" in response.headers:
                                content_disp = response.headers["Content-Disposition"]
                                if "filename=" in content_disp:
                                    filename = content_disp.split("filename=")[1].strip('"')
                            
                            st.session_state.document_filename = filename
                            st.session_state.document_type = "Writ Petition"
                            st.success("Writ Petition generated successfully!")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
        
        # Display download button outside the form if document is generated
        if st.session_state.document_response is not None and st.session_state.document_type == "Writ Petition":
            st.download_button(
                label="Download Writ Petition Document",
                data=st.session_state.document_response.content,
                file_name=st.session_state.document_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            # Add a button to clear the current document
            if st.button("Generate Another Writ Petition"):
                st.session_state.document_response = None
                st.session_state.document_filename = None
                st.session_state.document_type = None
                st.rerun()
    
    # Affidavit Form
    elif selected_doc == "Affidavit":
        with st.form("affidavit_form"):
            st.markdown("<div class='form-container'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                case_number = st.text_input("Case Number:")
                year = st.number_input("Year:", min_value=1900, max_value=2100, value=2023)
                petitioner = st.text_input("Petitioner:")
                respondent = st.text_input("Respondent:")
            
            with col2:
                affiant_name = st.text_input("Affiant Name:")
                order_date = st.text_input("Order Date:")
                property_name = st.text_input("Property Name:")
            
            statement_of_facts = st.text_area("Statement of Facts:")
            
            submit = st.form_submit_button("Generate Affidavit")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if submit:
                # Create request data
                request_data = AffidavitRequest(
                    case_number=case_number,
                    year=year,
                    petitioner=petitioner,
                    respondent=respondent,
                    affiant_name=affiant_name,
                    order_date=order_date,
                    property_name=property_name,
                    statement_of_facts=statement_of_facts
                )
                
                # Send API request for Affidavit
                with st.spinner("Generating Affidavit..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/generate/affidavit",
                            json=request_data.model_dump()
                        )
                        if response.status_code == 200:
                            st.session_state.document_response = response
                            
                            # Get filename from response headers if available
                            filename = "Affidavit.docx"
                            if "Content-Disposition" in response.headers:
                                content_disp = response.headers["Content-Disposition"]
                                if "filename=" in content_disp:
                                    filename = content_disp.split("filename=")[1].strip('"')
                            
                            st.session_state.document_filename = filename
                            st.session_state.document_type = "Affidavit"
                            st.success("Affidavit generated successfully!")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
        
        # Display download button outside the form if document is generated
        if st.session_state.document_response is not None and st.session_state.document_type == "Affidavit":
            st.download_button(
                label="Download Affidavit Document",
                data=st.session_state.document_response.content,
                file_name=st.session_state.document_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            # Add a button to clear the current document
            if st.button("Generate Another Affidavit"):
                st.session_state.document_response = None
                st.session_state.document_filename = None
                st.session_state.document_type = None
                st.rerun()
    
    # Patent Application Form
    elif selected_doc == "Patent Application":
        with st.form("patent_application_form"):
            st.markdown("<div class='form-container'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                application_number = st.text_input("Application Number:")
                year = st.number_input("Year:", min_value=1900, max_value=2100, value=2023)
                inventor_name = st.text_input("Inventor Name:")
                assignee = st.text_input("Assignee:")
            
            with col2:
                title = st.text_input("Title:")
                field_of_invention = st.text_input("Field of Invention:")
                
            abstract = st.text_area("Abstract:")
            background = st.text_area("Background:")
            summary = st.text_area("Summary:")
            claims = st.text_area("Claims:")
            drawings_description = st.text_area("Drawings Description:")
            
            submit = st.form_submit_button("Generate Patent Application")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if submit:
                # Create request data
                request_data = PatentApplicationRequest(
                    application_number=application_number,
                    abstract=abstract,
                    year=year,
                    inventor_name=inventor_name,
                    assignee=assignee,
                    title=title,
                    field_of_invention=field_of_invention,
                    background=background,
                    summary=summary,
                    claims=claims,
                    drawings_description=drawings_description
                )
                
                # Send API request for Patent Application
                with st.spinner("Generating Patent Application..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/generate/patent_application",
                            json=request_data.model_dump()
                        )
                        if response.status_code == 200:
                            st.session_state.document_response = response
                            
                            # Get filename from response headers if available
                            filename = "Patent_Application.docx"
                            if "Content-Disposition" in response.headers:
                                content_disp = response.headers["Content-Disposition"]
                                if "filename=" in content_disp:
                                    filename = content_disp.split("filename=")[1].strip('"')
                            
                            st.session_state.document_filename = filename
                            st.session_state.document_type = "Patent Application"
                            st.success("Patent Application generated successfully!")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
        
        # Display download button outside the form if document is generated
        if st.session_state.document_response is not None and st.session_state.document_type == "Patent Application":
            st.download_button(
                label="Download Patent Application Document",
                data=st.session_state.document_response.content,
                file_name=st.session_state.document_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            # Add a button to clear the current document
            if st.button("Generate Another Patent Application"):
                st.session_state.document_response = None
                st.session_state.document_filename = None
                st.session_state.document_type = None
                st.rerun()
    
    # Annexure Form
    elif selected_doc == "Annexure":
        with st.form("annexure_form"):
            st.markdown("<div class='form-container'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                case_number = st.text_input("Case Number:")
                year = st.number_input("Year:", min_value=1900, max_value=2100, value=2023)
                petitioner = st.text_input("Petitioner:")
                respondent = st.text_input("Respondent:")
            
            with col2:
                annexure_title = st.text_input("Annexure Title:")
                annexure_number = st.text_input("Annexure Number:")
            
            description = st.text_area("Description:")
            supporting_docs = st.text_area("Supporting Documents (one per line):")
            supporting_documents = [doc.strip() for doc in supporting_docs.split("\n") if doc.strip()]
            
            submit = st.form_submit_button("Generate Annexure")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if submit:
                # Create request data
                request_data = AnnexureRequest(
                    case_number=case_number,
                    year=year,
                    petitioner=petitioner,
                    respondent=respondent,
                    annexure_title=annexure_title,
                    annexure_number=annexure_number,
                    description=description,
                    supporting_documents=supporting_documents
                )
                
                # Send API request for Annexure
                with st.spinner("Generating Annexure..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/generate/annexure",
                            json=request_data.model_dump()
                        )
                        if response.status_code == 200:
                            st.session_state.document_response = response
                            
                            # Get filename from response headers if available
                            filename = "Annexure.docx"
                            if "Content-Disposition" in response.headers:
                                content_disp = response.headers["Content-Disposition"]
                                if "filename=" in content_disp:
                                    filename = content_disp.split("filename=")[1].strip('"')
                            
                            st.session_state.document_filename = filename
                            st.session_state.document_type = "Annexure"
                            st.success("Annexure generated successfully!")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
        
        # Display download button outside the form if document is generated
        if st.session_state.document_response is not None and st.session_state.document_type == "Annexure":
            st.download_button(
                label="Download Annexure Document",
                data=st.session_state.document_response.content,
                file_name=st.session_state.document_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            # Add a button to clear the current document
            if st.button("Generate Another Annexure"):
                st.session_state.document_response = None
                st.session_state.document_filename = None
                st.session_state.document_type = None
                st.rerun()
    
    # Witness Statement Form
    elif selected_doc == "Witness Statement":
        with st.form("witness_statement_form"):
            st.markdown("<div class='form-container'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                case_number = st.text_input("Case Number:")
                year = st.number_input("Year:", min_value=1900, max_value=2100, value=2023)
                court_name = st.text_input("Court Name:")
            
            with col2:
                witness_name = st.text_input("Witness Name:")
                witness_details = st.text_area("Witness Details:")
            
            statement = st.text_area("Statement:")
            
            submit = st.form_submit_button("Generate Witness Statement")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if submit:
                # Create request data
                request_data = WitnessStatementRequest(
                    case_number=case_number,
                    year=year,
                    court_name=court_name,
                    witness_name=witness_name,
                    witness_details=witness_details,
                    statement=statement
                )
                
                # Send API request for Witness Statement
                with st.spinner("Generating Witness Statement..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/generate/witness_statement",
                            json=request_data.model_dump()
                        )
                        if response.status_code == 200:
                            st.session_state.document_response = response
                            
                            # Get filename from response headers if available
                            filename = "Witness_Statement.docx"
                            if "Content-Disposition" in response.headers:
                                content_disp = response.headers["Content-Disposition"]
                                if "filename=" in content_disp:
                                    filename = content_disp.split("filename=")[1].strip('"')
                            
                            st.session_state.document_filename = filename
                            st.session_state.document_type = "Witness Statement"
                            st.success("Witness Statement generated successfully!")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
        
        # Display download button outside the form if document is generated
        if st.session_state.document_response is not None and st.session_state.document_type == "Witness Statement":
            st.download_button(
                label="Download Witness Statement Document",
                data=st.session_state.document_response.content,
                file_name=st.session_state.document_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            # Add a button to clear the current document
            if st.button("Generate Another Witness Statement"):
                st.session_state.document_response = None
                st.session_state.document_filename = None
                st.session_state.document_type = None
                st.rerun()
    
    # Exhibit Form
    elif selected_doc == "Exhibit":
        with st.form("exhibit_form"):
            st.markdown("<div class='form-container'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                case_number = st.text_input("Case Number:")
                year = st.number_input("Year:", min_value=1900, max_value=2100, value=2023)
            
            with col2:
                exhibit_number = st.text_input("Exhibit Number:")
                exhibit_title = st.text_input("Exhibit Title:")
            
            description = st.text_area("Description:")
            attached_docs = st.text_area("Attached Documents (one per line):")
            attached_documents = [doc.strip() for doc in attached_docs.split("\n") if doc.strip()]
            
            submit = st.form_submit_button("Generate Exhibit")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if submit:
                # Create request data
                request_data = ExhibitRequest(
                    case_number=case_number,
                    year=year,
                    exhibit_number=exhibit_number,
                    exhibit_title=exhibit_title,
                    description=description,
                    attached_documents=attached_documents
                )
                
                # Send API request for Exhibit
                with st.spinner("Generating Exhibit..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/generate/exhibit",
                            json=request_data.model_dump()
                        )
                        if response.status_code == 200:
                            st.session_state.document_response = response
                            
                            # Get filename from response headers if available
                            filename = "Exhibit.docx"
                            if "Content-Disposition" in response.headers:
                                content_disp = response.headers["Content-Disposition"]
                                if "filename=" in content_disp:
                                    filename = content_disp.split("filename=")[1].strip('"')
                            
                            st.session_state.document_filename = filename
                            st.session_state.document_type = "Exhibit"
                            st.success("Exhibit generated successfully!")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
        
        # Display download button outside the form if document is generated
        if st.session_state.document_response is not None and st.session_state.document_type == "Exhibit":
            st.download_button(
                label="Download Exhibit Document",
                data=st.session_state.document_response.content,
                file_name=st.session_state.document_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            # Add a button to clear the current document
            if st.button("Generate Another Exhibit"):
                st.session_state.document_response = None
                st.session_state.document_filename = None
                st.session_state.document_type = None
                st.rerun()
    
    # Forensic Report Form
    elif selected_doc == "Forensic Report":
        with st.form("forensic_report_form"):
            st.markdown("<div class='form-container'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                case_number = st.text_input("Case Number:")
                year = st.number_input("Year:", min_value=1900, max_value=2100, value=2023)
            
            with col2:
                forensic_expert = st.text_input("Forensic Expert:")
                forensic_field = st.text_input("Forensic Field:")
            
            report_summary = st.text_area("Report Summary:")
            findings = st.text_area("Findings:")
            conclusion = st.text_area("Conclusion:")
            
            submit = st.form_submit_button("Generate Forensic Report")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if submit:
                # Create request data
                request_data = ForensicReportRequest(
                    case_number=case_number,
                    year=year,
                    forensic_expert=forensic_expert,
                    forensic_field=forensic_field,
                    report_summary=report_summary,
                    findings=findings,
                    conclusion=conclusion
                )
                
                # Send API request for Forensic Report
                with st.spinner("Generating Forensic Report..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/generate/forensic_report",
                            json=request_data.model_dump()
                        )
                        if response.status_code == 200:
                            st.session_state.document_response = response
                            
                            # Get filename from response headers if available
                            filename = "Forensic_Report.docx"
                            if "Content-Disposition" in response.headers:
                                content_disp = response.headers["Content-Disposition"]
                                if "filename=" in content_disp:
                                    filename = content_disp.split("filename=")[1].strip('"')
                            
                            st.session_state.document_filename = filename
                            st.session_state.document_type = "Forensic Report"
                            st.success("Forensic Report generated successfully!")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
        
        # Display download button outside the form if document is generated
        if st.session_state.document_response is not None and st.session_state.document_type == "Forensic Report":
            st.download_button(
                label="Download Forensic Report Document",
                data=st.session_state.document_response.content,
                file_name=st.session_state.document_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            # Add a button to clear the current document
            if st.button("Generate Another Forensic Report"):
                st.session_state.document_response = None
                st.session_state.document_filename = None
                st.session_state.document_type = None
                st.rerun()
    
    # Expert Opinion Form
    elif selected_doc == "Expert Opinion":
        with st.form("expert_opinion_form"):
            st.markdown("<div class='form-container'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                case_number = st.text_input("Case Number:")
                year = st.number_input("Year:", min_value=1900, max_value=2100, value=2023)
            
            with col2:
                expert_name = st.text_input("Expert Name:")
                field_of_expertise = st.text_input("Field of Expertise:")
            
            opinion_summary = st.text_area("Opinion Summary:")
            detailed_opinion = st.text_area("Detailed Opinion:")
            supporting_refs = st.text_area("Supporting References (one per line):")
            supporting_references = [ref.strip() for ref in supporting_refs.split("\n") if ref.strip()]
            
            submit = st.form_submit_button("Generate Expert Opinion")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if submit:
                # Create request data
                request_data = ExpertOpinionRequest(
                    case_number=case_number,
                    year=year,
                    expert_name=expert_name,
                    field_of_expertise=field_of_expertise,
                    opinion_summary=opinion_summary,
                    detailed_opinion=detailed_opinion,
                    supporting_references=supporting_references
                )
                
                # Send API request for Expert Opinion
                with st.spinner("Generating Expert Opinion..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/generate/expert_opinion",
                            json=request_data.model_dump()
                        )
                        if response.status_code == 200:
                            st.session_state.document_response = response
                            
                            # Get filename from response headers if available
                            filename = "Expert_Opinion.docx"
                            if "Content-Disposition" in response.headers:
                                content_disp = response.headers["Content-Disposition"]
                                if "filename=" in content_disp:
                                    filename = content_disp.split("filename=")[1].strip('"')
                            
                            st.session_state.document_filename = filename
                            st.session_state.document_type = "Expert Opinion"
                            st.success("Expert Opinion generated successfully!")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
        
        # Display download button outside the form if document is generated
        if st.session_state.document_response is not None and st.session_state.document_type == "Expert Opinion":
            st.download_button(
                label="Download Expert Opinion Document",
                data=st.session_state.document_response.content,
                file_name=st.session_state.document_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            # Add a button to clear the current document
            if st.button("Generate Another Expert Opinion"):
                st.session_state.document_response = None
                st.session_state.document_filename = None
                st.session_state.document_type = None
                st.rerun()

# Knowledge Base functionality
elif st.session_state.current_tab == "Knowledge Base":
    st.markdown("<div class='sub-header'>Knowledge Base Management</div>", unsafe_allow_html=True)
    
    # First display available namespaces
    st.markdown("### Available Document Collections")
    namespaces = chatbot.list_namespaces()
    if namespaces:
        # Create a dataframe for better display
        namespace_list = []
        for ns in namespaces:
            namespace_list.append({"Document Name": ns})
        
        # Display as a table with improved styling
        st.dataframe(
            namespace_list,
            column_config={"Document Name": st.column_config.TextColumn("Document Name")},
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No document collections found in the database. Upload documents below to create your knowledge base.")
    
    # Separator
    st.markdown("---")
    
    # Add new documents section
    st.markdown("### Add Documents to Knowledge Base")
    st.markdown("""
    Upload PDF files to add them to your knowledge base. 
    Once processed, these documents will be available for querying in the Chatbot.
    """)
    
    # Simplified file uploader
    uploaded_files = st.file_uploader("Upload PDF Document(s)", type="pdf", accept_multiple_files=True)
    
    # Custom namespace options
    namespace_option = st.radio(
        "Namespace Options", 
        ["Use filename as namespace", "Use custom namespace", "Add prefix to filenames"],
        horizontal=True
    )
    
    custom_input = ""
    if namespace_option == "Use custom namespace":
        custom_input = st.text_input(
            "Custom Namespace", 
            help="Enter a custom identifier for all documents. This will be used instead of the filename."
        )
    elif namespace_option == "Add prefix to filenames":
        custom_input = st.text_input(
            "Namespace Prefix", 
            help="Enter a prefix for all documents. Each file will use this prefix plus its filename."
        )
    
    # Process uploaded files
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} file(s):")
        for file in uploaded_files:
            st.write(f"- {file.name}")
        
        if st.button("Process Document(s)", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing files..."):
                results = []
                total_files = len(uploaded_files)
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing file {i+1}/{total_files}: {file.name}")
                    progress_bar.progress((i) / total_files)
                    
                    # Determine namespace based on selection
                    if namespace_option == "Use filename as namespace":
                        custom_ns = None
                    elif namespace_option == "Use custom namespace":
                        custom_ns = custom_input
                    else:  # Add prefix
                        custom_ns = f"{custom_input}_{file.name}" if custom_input else file.name
                        
                    result = process_uploaded_file(file, custom_ns)
                    results.append(result)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / total_files)
                
                # Save results in session state
                st.session_state.upload_status = {
                    "results": results,
                    "total": len(results),
                    "successful": sum(1 for r in results if r["status"] == "success"),
                    "failed": sum(1 for r in results if r["status"] == "error")
                }
                
                # Complete the progress bar
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
    
    # Display upload status 
    if st.session_state.upload_status:
        st.subheader("Processing Results")
        
        results = st.session_state.upload_status["results"]
        successful = st.session_state.upload_status["successful"]
        failed = st.session_state.upload_status["failed"]
        
        st.write(f"Processed {len(results)} files: {successful} successful, {failed} failed")
        
        if successful > 0:
            with st.expander("View Successfully Processed Files", expanded=True):
                for result in results:
                    if result["status"] == "success":
                        st.success(f"""
                        - Filename: {result["filename"]}
                        - Namespace: {result["namespace"]}
                        - Pages: {result["pages"]}
                        - Chunks: {result["chunks"]}
                        """)
        
        if failed > 0:
            with st.expander("View Failed Files", expanded=True):
                for result in results:
                    if result["status"] == "error":
                        st.error(f"""
                        - Filename: {result["filename"]}
                        - Error: {result["error"]}
                        """)
        
        if st.button("Upload More Documents"):
            st.session_state.upload_status = None
            st.rerun() 