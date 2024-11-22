from langchain_community.llms import Ollama
from flask import Flask, request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import os
from flask import request, jsonify
from langchain_core.prompts import ChatPromptTemplate

from langchain.docstore.document import Document
from PIL import Image
import pytesseract
from transformers import LlamaTokenizer, LlamaConfig

pytesseract.pytesseract.tesseract_cmd = "/home/s188903/tesseract/bin/tesseract"


from pdf2image import convert_from_path
import os
os.environ['PATH'] = os.path.expanduser("~/poppler/bin") + ":" + os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = os.path.expanduser("~/poppler/lib") + ":" + os.environ.get('LD_LIBRARY_PATH', '')
from transformers import LlamaTokenizer, LlamaConfig


def process_pdf_with_ocr(file_path):
    # Konwertuj strony PDF na obrazy
    pages = convert_from_path(file_path)
    docs = []
    for i, page in enumerate(pages):
        # Przetwarzaj każdą stronę OCR
        text = pytesseract.image_to_string(page, lang='eng')
        docs.append(Document(page_content=text, metadata={"source": file_path, "page": i + 1}))
    return docs


app = Flask(__name__)

folder_path = "db"

cached_llm = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()

text_splitter = CharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False)


raw_prompt_template = (
"Please answer the following question based on the provided context. Be precise and go straight to the point. If you don't have an answer from the provided information, please say so."
"{context}"
)

raw_prompt = PromptTemplate(
    template=raw_prompt_template,
    input_variables=["question", "context"]
)
prompt = ChatPromptTemplate.from_messages(
        [
        ("system", raw_prompt_template),
        ("human", "{input}"),
    ]
)


@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    response = cached_llm.invoke(query)
    print(response)
    response_answer = {"answer": response}
    return response_answer

from langchain.chains import RetrievalQA

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    print('Loading vector store')
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print('Creating retriever')
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': 1,
        }
    )

    print('Setting up the LLM and chain')
    document_chain = create_stuff_documents_chain(cached_llm, prompt)
    chain = create_retrieval_chain(retriever,document_chain)
    result = chain.invoke({"input":query})
    print(f"Retrieved documentsXXX: {result['context']}")
    print(result)
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=cached_llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     chain_type_kwargs={
    #         "prompt": raw_prompt + query
    #     },
    #     return_source_documents=True  # This ensures that source documents are returned
    # )

    # print('Running the chain')
    # result = qa_chain({"query": query})
    # print(f"Result: {result}")

    # Extract the answer and sources
    #answer = result["result"]
    #source_documents = result["source_documents"]
    source_documents = result["context"]

    # Prepare sources for the response
    sources = []
    for doc in source_documents:
        sources.append({
            "source": doc.metadata.get("source", ""),
            "page_context": doc.page_content
        })

    response_answer = {"answer": result["answer"], "sources": sources}
    return jsonify(response_answer)



from langchain.docstore.document import Document  # Import klasy Document

@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    # Initialize the database
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    # Check if the document already exists in the database
    existing_docs = vector_store.get(where={'source': save_file})['ids']
    if existing_docs:
        return {
            "status": "error",
            "message": f"Document '{file_name}' already exists in the database."
        }, 409  # HTTP 409 Conflict

    # Try extracting text directly
    try:
        print("Attempting to load and split document without OCR...")
        loader = PDFPlumberLoader(save_file)
        docs = loader.load_and_split()
        combined_text = "".join(doc.page_content for doc in docs)

        if not combined_text.strip():
            raise ValueError("No text found in document. Falling back to OCR.")

    except Exception as e:
        print(f"Direct text extraction failed: {e}")
        print("Falling back to OCR...")
        docs = process_pdf_with_ocr(save_file)

    print(f"docs len={len(docs)}")

    # Split documents into smaller chunks
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    # Add documents with metadata to the database
    documents_with_metadata = [
        Document(page_content=chunk.page_content, metadata={"source": save_file}) for chunk in chunks
    ]
    vector_store.add_documents(documents_with_metadata)

    vector_store.persist()  # Save changes

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response




@app.route("/pdf_sources", methods=["GET"])
def get_pdf_sources():
    print("Getting PDF sources")
    context_window_size = config.max_position_embeddings
    print(f"Context window size: {context_window_size} tokens")
    # Ładowanie bazy danych Chroma
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    
    # Pobranie dokumentów z bazy danych
    docs = vector_store.similarity_search("dummy query", k=1000)  # 'dummy query' ponieważ potrzebujemy losowych dokumentów
    
    # Sprawdzenie źródeł
    sources = []
    for doc in docs:
        sources.append(
            {"source": doc.metadata["source"]}
        )
    
    response_answer = {"sources": sources, "number_of_documents": len(docs)}
    return response_answer


@app.route("/delete_pdf_by_source", methods=["POST"])
def delete_pdf_by_source():
    json_content = request.json
    source_name = json_content.get("source")  # np. "pdf/faktura_206216343.pdf"
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    try:
        # Pobierz identyfikatory dokumentów o określonym "source"
        ids = vector_store.get(where={'source': source_name})['ids']
        
        # Usuń dokumenty o znalezionych identyfikatorach
        if ids:
            vector_store.delete(ids)
            message = f"Document with source '{source_name}' deleted successfully."
        else:
            message = f"No document found with source '{source_name}'."

        return {"status": "success", "message": message}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


def start_app():
    app.run(host = "0.0.0.0", port=2137, debug=True)

if __name__ == "__main__":
    start_app()
