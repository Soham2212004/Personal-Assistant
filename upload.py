import os
import PyPDF2
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import uuid

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "rag-chatbot")
PDF_PATH = os.getenv("PDF_PATH")  # Path to your PDF file

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def get_embeddings(text):
    """Get embeddings using Gemini"""
    try:
        model = 'models/text-embedding-004'
        response = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None

def create_index_if_not_exists():
    """Create Pinecone index if it doesn't exist"""
    try:
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            print(f"Creating index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,  # Gemini text-embedding-004 dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print(f"Index {INDEX_NAME} created successfully!")
        else:
            print(f"Index {INDEX_NAME} already exists.")
            
        return pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Error creating index: {e}")
        return None

def upload_document_to_pinecone():
    """Main function to upload document to Pinecone"""
    
    if not PDF_PATH or not os.path.exists(PDF_PATH):
        print("Please provide a valid PDF_PATH in your .env file")
        return
    
    print("Starting document upload process...")
    
    # Extract text from PDF
    print("Extracting text from PDF...")
    document_text = extract_text_from_pdf(PDF_PATH)
    
    if not document_text:
        print("Failed to extract text from PDF")
        return
    
    print(f"Extracted {len(document_text)} characters from PDF")
    
    # Split text into chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    
    chunks = text_splitter.split_text(document_text)
    print(f"Created {len(chunks)} text chunks")
    
    # Create or get index
    index = create_index_if_not_exists()
    if not index:
        print("Failed to create/get index")
        return
    
    # Process and upload chunks
    print("Generating embeddings and uploading to Pinecone...")
    
    vectors_to_upsert = []
    batch_size = 100
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 10:  # Skip very short chunks
            continue
            
        # Get embeddings
        embedding = get_embeddings(chunk)
        if not embedding:
            print(f"Failed to get embedding for chunk {i}")
            continue
        
        # Create unique ID for this chunk
        chunk_id = str(uuid.uuid4())
        
        # Prepare vector data
        vector_data = {
            'id': chunk_id,
            'values': embedding,
            'metadata': {
                'text': chunk,
                'chunk_index': i,
                'source': os.path.basename(PDF_PATH)
            }
        }
        
        vectors_to_upsert.append(vector_data)
        
        # Upload in batches
        if len(vectors_to_upsert) >= batch_size:
            try:
                index.upsert(vectors=vectors_to_upsert)
                print(f"Uploaded batch of {len(vectors_to_upsert)} vectors")
                vectors_to_upsert = []
            except Exception as e:
                print(f"Error uploading batch: {e}")
    
    # Upload remaining vectors
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            print(f"Uploaded final batch of {len(vectors_to_upsert)} vectors")
        except Exception as e:
            print(f"Error uploading final batch: {e}")
    
    print("Document upload completed!")
    
    # Get index stats
    try:
        stats = index.describe_index_stats()
        print(f"Index now contains {stats['total_vector_count']} vectors")
    except Exception as e:
        print(f"Error getting index stats: {e}")

if __name__ == "__main__":
    upload_document_to_pinecone()
