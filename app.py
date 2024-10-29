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

app = Flask(__name__)

folder_path = "db"

cached_llm = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()

text_splitter = CharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False)

raw_prompt = PromptTemplate.from_template("""
    <s>[INST] Please help me answer the following question. If you do not have answer from provided information say so. Try to be precise and don't start with redundant introductory lines, get streight to the point. [/INST] </s>               
    [INST] {input} 
            Context: {context}
            Answer:                                                                       
                                [/INST]          """)

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

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    print('Loading vector store')
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print('Creatinig chain')
    retreiver = vector_store.as_retriever(
        search_type = 'similarity_score_threshold',
        search_kwargs = {
            'k':5,
            'score_threshold':0.5
        }
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retreiver,document_chain)

    result = chain.invoke({"input":query})
    print(f"Retrieved documentsXXX: {result['context']}")
    print(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_context": doc.page_content}
        )
    
    docs = vector_store.similarity_search("dummy query", k=10)
    print(f"Number of documents in the vector store: {len(docs)}")


    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


from langchain.docstore.document import Document  # Import klasy Document

@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    # Inicjalizacja bazy danych
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    # Sprawdzenie, czy dokument o tym source już istnieje
    existing_docs = vector_store.get(where={'source': save_file})['ids']
    if existing_docs:
        return {
            "status": "error",
            "message": f"Document '{file_name}' already exists in the database."
        }, 409  # HTTP 409 Conflict

    # Jeśli dokument nie istnieje, przetwórz i dodaj do bazy
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    # Tworzenie obiektów Document z metadanymi dla każdego chunku
    documents_with_metadata = [Document(page_content=chunk.page_content, metadata={"source": save_file}) for chunk in chunks]
    
    # Dodanie dokumentów do bazy
    vector_store.add_documents(documents_with_metadata)

    vector_store.persist()  # Zapisz zmiany

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
