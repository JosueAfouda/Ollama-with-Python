import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema import Document
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

def ingest_pdf(doc_path: str):
    """Load PDF documents using UnstructuredPDFLoader."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info(f"PDF loaded successfully from {doc_path}.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None

def ingest_pdf_plumber(doc_path: str):
    """Load PDF documents using PDFPlumberLoader."""
    if os.path.exists(doc_path):
        loader = PDFPlumberLoader(file_path=doc_path)
        pages = loader.load_and_split()
        logging.info(f"PDF loaded successfully from {doc_path} using PDFPlumberLoader.")
        return pages
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None

def split_documents(documents, chunk_size=1200, chunk_overlap=300):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Documents split into {len(chunks)} chunks.")
    return chunks

def split_text_into_chunks(text_content: str, chunk_size=1200, chunk_overlap=300):
    """Split raw text content into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text_content)
    logging.info(f"Text content split into {len(chunks)} chunks.")
    return chunks

def create_vector_db(chunks, embedding_model_name="nomic-embed-text", collection_name="simple-rag", persist_directory=None, use_fastembed=False):
    """Create a vector database from document chunks."""
    if use_fastembed:
        embedding = FastEmbedEmbeddings()
        logging.info("Using FastEmbedEmbeddings.")
    else:
        ollama.pull(embedding_model_name)
        embedding = OllamaEmbeddings(model=embedding_model_name)
        logging.info(f"Using OllamaEmbeddings with model: {embedding_model_name}.")

    if persist_directory and os.path.exists(persist_directory):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        logging.info(f"Loaded existing vector database from {persist_directory}.")
    else:
        docs = [
            Document(page_content=chunk.page_content, metadata=chunk.metadata)
            if hasattr(chunk, 'page_content') else Document(page_content=chunk)
            for chunk in chunks
        ]
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        if persist_directory:
            vector_db.persist()
            logging.info(f"Vector database created and persisted to {persist_directory}.")
        else:
            logging.info("Vector database created (in-memory).")
    return vector_db
