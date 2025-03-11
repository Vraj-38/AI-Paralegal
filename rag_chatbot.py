import os
import faiss
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import glob

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize LLM
        self.llm = ChatGroq(
            model_name="mixtral-8x7b-32768",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3
        )
        
        # Initialize vector store
        self.vector_store = None
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )

        # Define general conversation patterns
        self.general_patterns = {
            "greeting": ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"],
            "how_are_you": ["how are you?", "how are you doing?", "how's it going?", "what's up?"],
            "what_do_you_do": ["what do you do?", "what are you?", "who are you?", "what is your purpose?", "what can you do?"],
            "goodbye": ["bye", "goodbye", "see you", "farewell", "take care"]
        }

    def is_general_query(self, query: str) -> tuple[bool, str]:
        """
        Check if the query is a general conversation query.
        Now matches exact phrases or greetings only.
        """
        query_lower = query.lower().strip().rstrip('?!.,')
        
        # First, check for exact matches
        for category, patterns in self.general_patterns.items():
            if query_lower in patterns:
                return True, category
            
        # For greetings and goodbyes, also check if query contains only these words
        if category in ["greeting", "goodbye"]:
            query_words = set(query_lower.split())
            for pattern in patterns:
                pattern_words = set(pattern.split())
                if query_words == pattern_words:
                    return True, category
        
        # Special case for longer queries
        if len(query_lower.split()) > 3:  # If query is longer than 3 words
            return False, ""  # Assume it's a specific question
            
        return False, ""

    def get_general_response(self, category: str) -> str:
        """Get response for general conversation queries."""
        responses = {
            "greeting": "Hello! I'm your AI Paralegal assistant. How can I help you today?",
            "how_are_you": "I'm functioning well and ready to assist you with legal research and document analysis. How can I help you?",
            "what_do_you_do": "I am an AI Paralegal assistant specialized in analyzing legal documents and precedents. I can help you understand legal documents, extract relevant information from case files, and answer questions based on legal precedents. Just upload your documents and I'll assist you with the analysis.",
            "goodbye": "Goodbye! Feel free to return if you need any assistance with legal document analysis."
        }
        return responses.get(category, "I'm here to help you analyze legal documents and answer your questions. Would you like to upload some documents?")

    def load_from_directory(self, directory_path: str):
        """Load all PDF documents from a directory."""
        # Normalize the directory path
        directory_path = os.path.normpath(directory_path)
        
        # Find all PDF files in the directory
        pdf_pattern = os.path.join(directory_path, '*.pdf')
        pdf_files = glob.glob(pdf_pattern)
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {directory_path}")
        
        # Sort files by name for consistent loading order
        pdf_files.sort()
        
        print(f"\nFound {len(pdf_files)} PDF files to process:")
        for file in pdf_files:
            print(f"- {os.path.basename(file)}")
            
        self.load_documents(pdf_files)

    def load_documents(self, file_paths: List[str]):
        """Load PDF documents and process them."""
        documents = []
        total_pages = 0
        
        print("\nLoading PDF files...")
        for file_path in file_paths:
            try:
                loader = PyPDFLoader(file_path)
                file_docs = loader.load()
                documents.extend(file_docs)
                
                print(f"✓ {os.path.basename(file_path)} - {len(file_docs)} pages")
                total_pages += len(file_docs)
                
            except Exception as e:
                print(f"✗ Error loading {os.path.basename(file_path)}: {str(e)}")
        
        if not documents:
            raise ValueError("No documents were successfully loaded")
        
        print(f"\nTotal pages loaded: {total_pages}")
        
        # Split documents into chunks
        print("\nProcessing documents...")
        texts = self.text_splitter.split_documents(documents)
        print(f"Created {len(texts)} text chunks")
        
        # Create vector store
        print("\nCreating vector store...")
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        print("Vector store created successfully!")

    def setup_qa_chain(self):
        """Setup the question-answering chain."""
        if not self.vector_store:
            raise ValueError("Please load documents first using load_documents()")
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )

    def chat(self, query: str):
        """Process a chat query and return the response."""
        # Only check for general queries if the query is short
        if len(query.split()) <= 3:  # Only check for general queries if 3 words or less
            is_general, category = self.is_general_query(query)
            if is_general:
                return {
                    "answer": self.get_general_response(category),
                    "sources": []  # No sources for general conversation
                }
        
        # If no documents are loaded and it's not a general query
        if not self.vector_store:
            return {
                "answer": "I notice you're asking about specific information, but no documents have been uploaded yet. Please upload some PDF documents so I can assist you better.",
                "sources": []
            }
        
        # Process document-based query
        qa_chain = self.setup_qa_chain()
        response = qa_chain.invoke({"query": query})
        
        return {
            "answer": response["result"],
            "sources": [
                {
                    "file": os.path.basename(doc.metadata.get("source", "Unknown")),
                    "page": doc.metadata.get("page", 1)
                }
                for doc in response["source_documents"]
            ]
        }

def main():
    # Example usage
    chatbot = RAGChatbot()
    
    # Get directory path from user
    directory_path = input("Enter the path to your PDFs folder: ")
    
    try:
        # Load documents from the specified directory
        chatbot.load_from_directory(directory_path)
        
        # Chat loop
        print("\nChatbot is ready! Type 'quit' to exit.")
        while True:
            query = input("\nYou: ")
            if query.lower() == 'quit':
                break
                
            try:
                response = chatbot.chat(query)
                print("\nBot:", response["answer"])
                print("\nSources:")
                for source in response["sources"]:
                    print(f"- {source['file']}, Page {source['page']}")
            except Exception as e:
                print(f"Error: {str(e)}")
    
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")

if __name__ == "__main__":
    main()
