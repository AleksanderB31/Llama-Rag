import argparse
from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from PIL import Image, ImageEnhance, ImageFilter
import os
import io
import fitz
import easyocr
import numpy as np
from transformers import GPT2Tokenizer


# Initialize Flask app
app = Flask(__name__)

def parse_arguments():
    """Parse command-line arguments for configuration."""
    parser = argparse.ArgumentParser(description="Flask Application Configuration")
    
    parser.add_argument(
    "--model_name",
    type=str,
    default="llama3_ctx_8192",
    help="Name of the LLM model being used."
    )

    parser.add_argument(
    "--num_of_retrieved_docs",
    type=int,
    default=8,
    help="Number of documents to retrieve in queries."
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=900,
        help="Maximum number of tokens per chunk."
    )
    
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=150,
        help="Number of overlapping tokens between chunks."
    )
    
    parser.add_argument(
    "--port",
    type=int,
    default=2137,
    help="Port number for the Flask app."
    )

    parser.add_argument(
        "--database_folder_path",
        type=str,
        default="db",
        help="Directory for the vector store."
    )

    parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser('~/fastembed_cache'),
    help="Directory for FastEmbed cache (default: ~/fastembed_cache)"
    ) 
    return parser.parse_args()

# Parse command-line arguments
args = parse_arguments()

# Configuration
CHUNK_SIZE = args.chunk_size
CHUNK_OVERLAP = args.chunk_overlap
DATABASE_FOLDER_PATH = args.database_folder_path
MODEL_NAME = args.model_name
NUM_OF_RETRIEVED_DOCS = args.num_of_retrieved_docs
PORT = args.port

# Initialize components
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
reader = easyocr.Reader(['en', 'pl'])
cached_llm = Ollama(model=MODEL_NAME)
embedding = FastEmbedEmbeddings(cache_dir=args.cache_dir)

# Prompt Templates
RAW_PROMPT_TEMPLATE = (
    "Please answer the following question based on the provided context. "
    "Be precise and go straight to the point. If you don't have an answer from the provided information, please say so.\n"
    "{context}"
)

raw_prompt = PromptTemplate(
    template=RAW_PROMPT_TEMPLATE,
    input_variables=["question", "context"]
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAW_PROMPT_TEMPLATE),
        ("human", "{input}"),
    ]
)

def token_count(text):
    """Count the number of tokens in the given text."""
    return len(tokenizer.encode(text, add_special_tokens=False))

def preprocess_image(image):
    """
    Preprocess the image by converting to grayscale, applying median filter,
    and enhancing contrast.
    
    :param image: PIL Image object.
    :return: Preprocessed PIL Image object.
    """
    # Convert to grayscale
    image = image.convert('L')
    # Apply median filter to reduce noise
    image = image.filter(ImageFilter.MedianFilter())
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    return image

def process_pdf_with_easyocr(file_path, zoom=2.0):
    """
    Process a PDF file using EasyOCR and PyMuPDF.

    :param file_path: Path to the PDF file.
    :param zoom: Zoom factor for rendering pages.
    :return: List of Document objects with recognized text.
    """
    # Open the PDF document
    doc = fitz.open(file_path)
    docs = []
    
    # Define zoom matrix
    zoom_matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Render page as pixmap with zoom
        pix = page.get_pixmap(matrix=zoom_matrix)
        
        # Convert pixmap to PIL Image
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Convert PIL Image to numpy array
        image_np = np.array(preprocessed_image)
        
        # Perform OCR on the numpy image
        result = reader.readtext(image_np, detail=0, paragraph=True)
        text = "\n".join(result)
        
        docs.append(Document(page_content=text, metadata={"source": file_path, "page": page_num + 1}))
    
    return docs

def extract_text_with_pymupdf(file_path):
    """
    Extract text from a PDF file using PyMuPDF.

    :param file_path: Path to the PDF file.
    :return: Extracted text as a single string.
    """
    doc = fitz.open(file_path)
    all_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        all_text += text + " "  # Add space between pages

    return all_text.strip()

def sliding_window_split(text, tokenizer, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split text into chunks with a sliding window approach based on token count using GPT-2 tokenizer.

    :param text: Full text to split.
    :param tokenizer: GPT-2 Tokenizer instance.
    :param chunk_size: Maximum number of tokens per chunk.
    :param chunk_overlap: Number of overlapping tokens between chunks.
    :return: List of text chunks.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end >= len(tokens) + 10:
            break
        start += (chunk_size - chunk_overlap)
    return chunks


@app.route("/ai", methods=["POST"])
def ai_post():
    """
    Endpoint to handle AI queries.

    Expects a JSON payload with a "query" field.
    Returns the AI-generated answer.
    """
    print("POST /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"Query: {query}")

    response = cached_llm.invoke(query)
    print(response)
    response_answer = {"answer": response}
    return response_answer

@app.route("/ask_pdf", methods=["POST"])
def ask_pdf_post():
    """
    Endpoint to handle PDF-based queries.

    Expects a JSON payload with a "query" field.
    Returns the AI-generated answer along with source documents.
    """
    print("POST /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")
    print(f"Query: {query}")

    print('Loading vector store')
    vector_store = Chroma(persist_directory=DATABASE_FOLDER_PATH, embedding_function=embedding)

    print('Creating retriever')
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': NUM_OF_RETRIEVED_DOCS,
        }
    )

    print('Setting up the LLM and chain')
    qa_chain = RetrievalQA.from_chain_type(
        llm=cached_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": raw_prompt + query
        },
        return_source_documents=True  # Ensure source documents are returned
    )

    print('Running the chain')
    result = qa_chain({"query": query})
    print(f"Result: {result}")

    # Extract the answer and sources
    answer = result["result"]
    source_documents = result["source_documents"]

    # Prepare sources for the response
    sources = [
        {
            "source": doc.metadata.get("source", ""),
            "page_context": doc.page_content
        }
        for doc in source_documents
    ]

    response_answer = {"answer": answer, "sources": sources}
    return jsonify(response_answer)

@app.route("/pdf", methods=["POST"])
def pdf_post():
    """
    Endpoint to upload and process a PDF file.

    Expects a file upload with the key "file".
    Returns the status of the upload and processing.
    """
    try:
        file = request.files["file"]
        file_name = file.filename
        save_dir = "pdf"
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, file_name)
        file.save(save_file)
        print(f"Filename: {file_name}")

        # Initialize vector store
        vector_store = Chroma(persist_directory=DATABASE_FOLDER_PATH, embedding_function=embedding)

        # Check if the document already exists in the database
        existing_docs = vector_store.get(where={'source': save_file})['ids']
        if existing_docs:
            return {
                "status": "error",
                "message": f"Document '{file_name}' already exists in the database."
            }, 409  # HTTP 409 Conflict

        # Attempt to extract text directly using PyMuPDF
        try:
            print("Attempting to load and split document without OCR using PyMuPDF...")
            combined_text = extract_text_with_pymupdf(save_file)
            if not combined_text.strip():
                raise ValueError("No text found in document.")

        except Exception as e:
            print(f"Direct text extraction with PyMuPDF failed: {e}")
            print("Proceeding with OCR...")
            combined_text = process_pdf_with_easyocr(save_file, zoom=3.0)
            if isinstance(combined_text, list):
                # If OCR was used, combined_text is a list of Document objects
                combined_text = "\n".join([doc.page_content for doc in combined_text])
            if not combined_text.strip():
                raise ValueError("Text extraction failed and OCR did not retrieve any text.")

        # Additional logging for diagnostics
        print(f"Combined text first 500 characters: {combined_text[:500]}")
        print(f"Total tokens in combined text: {len(tokenizer.encode(combined_text, add_special_tokens=False))}")

        # Split text into smaller chunks based on tokens
        chunks = sliding_window_split(combined_text, tokenizer, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"Number of chunks: {len(chunks)}")

        # Add documents with metadata to the vector store
        documents_with_metadata = [
            Document(
                page_content=chunk,
                metadata={"source": save_file, "chunk": i}
            )
            for i, chunk in enumerate(chunks, 1)
        ]

        vector_store.add_documents(documents_with_metadata)

        # Verify the number of tokens in each chunk
        for chunk in documents_with_metadata:
            num_tokens = token_count(chunk.page_content)
            print(f"Chunk has {num_tokens} tokens")
            assert num_tokens <= CHUNK_SIZE, "Chunk exceeds the token limit!"

        response = {
            "status": "Successfully Uploaded",
            "filename": file_name,
            "doc_len": len(documents_with_metadata),
            "chunks": len(documents_with_metadata)
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return {"status": "error", "message": str(e)}, 500

@app.route("/pdf_sources", methods=["GET"])
def get_pdf_sources():
    """
    Endpoint to retrieve all unique PDF sources from the vector store.

    Returns a list of sources and the number of documents.
    """
    print("Getting PDF sources")
    # Load Chroma vector store
    vector_store = Chroma(persist_directory=DATABASE_FOLDER_PATH, embedding_function=embedding)
    
    # Retrieve documents using a dummy query to get a broad set
    docs = vector_store.similarity_search("dummy query", k=1000)  # 'dummy query' to retrieve random documents
    
    # Collect unique sources by removing duplicates
    unique_sources = {doc.metadata["source"] for doc in docs}

    # Create a list of unique source dictionaries
    sources = [{"source": source} for source in unique_sources]
    
    response_answer = {"sources": sources, "number_of_documents": len(sources)}
    return response_answer

@app.route("/delete_pdf_by_source", methods=["POST"])
def delete_pdf_by_source():
    """
    Endpoint to delete PDF documents from the vector store based on the source name.

    Expects a JSON payload with a "source" field.
    Returns the status of the deletion.
    """
    json_content = request.json
    source_name = json_content.get("source")  
    vector_store = Chroma(persist_directory=DATABASE_FOLDER_PATH, embedding_function=embedding)

    try:
        # Retrieve document IDs with the specified source
        ids = vector_store.get(where={'source': source_name})['ids']
        
        # Delete documents with the retrieved IDs
        if ids:
            vector_store.delete(ids)
            message = f"Document with source '{source_name}' deleted successfully."
        else:
            message = f"No document found with source '{source_name}'."

        return {"status": "success", "message": message}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

def start_app():
    """Start the Flask application."""
    app.run(host="0.0.0.0", port=PORT, debug=True)

if __name__ == "__main__":
    start_app()
