import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pinecone
from langchain.docstore.document import Document
from tqdm import tqdm
from dotenv import load_dotenv
import time
import glob
import pandas as pd
import tempfile
import shutil

# Load environment variables from .env file
load_dotenv()

# === CONFIGURATION ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "ipd"

# Set static directory path for all PDF files to embed
EMBEDDING_DIRECTORY = r"D:\ipd\judmenents"

# Create directory if it doesn't exist
os.makedirs(EMBEDDING_DIRECTORY, exist_ok=True)

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# === STEP 1: Initialize Pinecone ===
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment='us-east1')

# === STEP 2: Initialize Google Embeddings ===
embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=GOOGLE_API_KEY)

# === STEP 3: Extract text from PDF ===
def extract_text_from_pdf(pdf_path):
    text_chunks = []
    doc = fitz.open(pdf_path)

    for page_num in tqdm(range(len(doc)), desc="Processing pages"):
        page = doc.load_page(page_num)
        direct_text = page.get_text()
        print(f"Page {page_num+1}: Extracted {len(direct_text)} chars")

        try:
            pix = page.get_pixmap(alpha=False)
            img_path = f"temp_page_{page_num}.png"
            pix.save(img_path)

            with Image.open(img_path) as img:
                ocr_text = pytesseract.image_to_string(img)

            combined_text = direct_text

            if ocr_text.strip():
                if ocr_text not in direct_text:
                    combined_text = combined_text + "\n\n--- OCR TEXT ---\n\n" + ocr_text
                    print(f"  Added OCR text ({len(ocr_text)} chars)")
                else:
                    print(f"  OCR text already included")

            os.remove(img_path)

            ocr_used = ocr_text.strip() != '' and ocr_text not in direct_text
            text = combined_text

        except Exception as e:
            print(f"  OCR failed: {str(e)}")
            ocr_used = False
            text = direct_text

        if not text.strip():
            print(f"  Warning: No text extracted from page {page_num+1}")

        text_chunks.append({
            "page": page_num + 1,
            "text": text,
            "ocr": ocr_used
        })

    return text_chunks

# === STEP 4: Chunk using LangChain ===
def chunk_text(text_data, source_name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    all_docs = []
    empty_pages = 0

    for entry in text_data:
        if not entry.get("text", "").strip():
            empty_pages += 1
            continue

        # For PDFs, we split the text
        chunks = splitter.split_text(entry["text"])
        print(f"Page {entry['page']}: Split into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source_name,
                    "page": entry["page"],
                    "chunk_id": i,
                    "ocr": entry.get("ocr", False),
                    "chunk_length": len(chunk),
                    "text": chunk,
                    "content_type": "pdf"
                }
            )
            all_docs.append(doc)

    if empty_pages > 0:
        print(f"Warning: {empty_pages} pages had no extractable content")

    return all_docs

# === STEP 5: Embed and upload to Pinecone ===
def upload_to_pinecone(docs, namespace):
    index = pc.Index(INDEX_NAME)

    if not docs:
        print("No documents to upload!")
        return

    print(f"Uploading {len(docs)} chunks to '{namespace}'")
    batch_size = 50

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        ids = [f"{doc.metadata['source']}-pdf-{doc.metadata.get('page', 0)}-c{doc.metadata['chunk_id']}" for doc in batch]
        texts = [doc.page_content for doc in batch]

        try:
            embeddings = embedder.embed_documents(texts)
            print(f"Batch {i//batch_size + 1}: {len(embeddings)} embeddings processed")

            metadatas = []
            for doc in batch:
                # Create metadata
                meta = {
                    "source": doc.metadata["source"],
                    "chunk_id": doc.metadata["chunk_id"],
                    "text": doc.page_content,
                    "content_type": "pdf",
                    "page": doc.metadata.get("page", 0),
                    "ocr": doc.metadata.get("ocr", False)
                }
                
                metadatas.append(meta)

            for j, meta in enumerate(metadatas):
                if not meta.get('text') or not meta['text'].strip():
                    print(f"Warning: Missing text content for chunk {meta.get('chunk_id', 'unknown')}")
                    meta['text'] = texts[j]

            vector_data = []
            for j in range(len(ids)):
                vector_data.append((ids[j], embeddings[j], metadatas[j]))

            index.upsert(vectors=vector_data, namespace=namespace)
            print(f"Batch {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1} uploaded")
            time.sleep(0.5)

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {str(e)}")
            for j, doc in enumerate(batch):
                try:
                    single_text = doc.page_content
                    single_embedding = embedder.embed_documents([single_text])[0]
                    single_id = ids[j]
                    
                    # Create metadata for single document
                    single_metadata = {
                        "source": doc.metadata["source"],
                        "chunk_id": doc.metadata["chunk_id"],
                        "text": single_text,
                        "content_type": "pdf",
                        "page": doc.metadata.get("page", 0),
                        "ocr": doc.metadata.get("ocr", False)
                    }
                    
                    index.upsert(vectors=[(single_id, single_embedding, single_metadata)], namespace=namespace)
                    print(f"  Uploaded individual doc {j}")
                    time.sleep(0.5)
                except Exception as inner_e:
                    print(f"  Failed on individual doc {j}: {str(inner_e)[:100]}")

    print(f"Completed upload to '{namespace}'")

# === RUN PDF PROCESSING ===
def process_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    print(f"\nProcessing PDF: {filename}")
    extracted = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(extracted)} pages")
    chunks = chunk_text(extracted, filename)
    print(f"Created {len(chunks)} chunks")
    upload_to_pinecone(chunks, namespace=filename)
    print(f"Completed processing PDF {filename}")

# === PROCESS ALL PDFs IN DIRECTORY ===
def process_all_pdfs():
    """Process all PDF files in the embedding directory"""
    print(f"Processing all PDF files in {EMBEDDING_DIRECTORY}...")
    
    # Get all PDF files
    pdf_files = glob.glob(os.path.join(EMBEDDING_DIRECTORY, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process each PDF
    for pdf_file in pdf_files:
        process_pdf(pdf_file)
    
    print("Completed processing all PDF files")

# === PROCESS UPLOADED PDF FILE ===
def process_uploaded_file(uploaded_file, custom_namespace=None):
    """
    Process a PDF file uploaded through a Streamlit interface
    
    Args:
        uploaded_file: The file object from st.file_uploader
        custom_namespace: Optional custom namespace name for Pinecone
        
    Returns:
        dict: Status information about the processing
    """
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Write the uploaded file content to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        filename = uploaded_file.name
        namespace = custom_namespace if custom_namespace else filename
        
        print(f"\nProcessing uploaded PDF: {filename}")
        extracted = extract_text_from_pdf(temp_file_path)
        print(f"Extracted {len(extracted)} pages")
        chunks = chunk_text(extracted, filename)
        print(f"Created {len(chunks)} chunks")
        upload_to_pinecone(chunks, namespace=namespace)
        print(f"Completed processing uploaded PDF {filename}")
        
        # Cleanup temporary file
        os.unlink(temp_file_path)
        
        return {
            "status": "success",
            "filename": filename,
            "namespace": namespace,
            "pages": len(extracted),
            "chunks": len(chunks)
        }
    except Exception as e:
        # Ensure temp file is cleaned up even if there's an error
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except:
            pass
        
        print(f"Error processing uploaded file: {str(e)}")
        return {
            "status": "error",
            "filename": uploaded_file.name,
            "error": str(e)
        }

# === PROCESS MULTIPLE UPLOADED FILES ===
def process_multiple_files(uploaded_files, namespace_prefix=None):
    """
    Process multiple PDF files uploaded through a Streamlit interface
    
    Args:
        uploaded_files: List of file objects from st.file_uploader
        namespace_prefix: Optional prefix for namespace names
        
    Returns:
        list: List of status dictionaries for each processed file
    """
    results = []
    
    for uploaded_file in uploaded_files:
        # Generate namespace with optional prefix
        if namespace_prefix:
            custom_namespace = f"{namespace_prefix}_{uploaded_file.name}"
        else:
            custom_namespace = None
            
        # Process the individual file
        result = process_uploaded_file(uploaded_file, custom_namespace)
        results.append(result)
        
        # Add a small delay between files to prevent rate limiting
        time.sleep(1)
    
    # Return results for all files
    return results

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("PDF Embedding Generator")
    print("=====================")
    print(f"Using directory: {EMBEDDING_DIRECTORY}")
    process_all_pdfs()