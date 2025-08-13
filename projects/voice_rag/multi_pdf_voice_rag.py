import os
import datetime
import asyncio
import logging
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import stream

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document

from src.utils.rag_utils import ingest_pdf_plumber, split_text_into_chunks, create_vector_db
from src.utils.ollama_utils import pull_ollama_model, generate_text_ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

model = "llama3.2"
vector_db_path = "./db/vector_db"
EMBEDDING_MODEL = "nomic-embed-text"

async def main():
    pdf_files = [f for f in os.listdir("./data") if f.endswith(".pdf")]

    all_pages = []

    for pdf_file in pdf_files:

        file_path = os.path.join("./data", pdf_file)
        logging.info(f"Processing PDF file: {pdf_file}")

        # Load the PDF file using utility function
        pages = ingest_pdf_plumber(file_path=file_path)
        if pages:
            logging.info(f"pages length: {len(pages)}")
            all_pages.extend(pages)

            # Extract text from the PDF file
            text = pages[0].page_content
            logging.info(f"Text extracted from the PDF file '{pdf_file}':\n{text}\n")

            # Prepare the prompt for the model
            prompt = f"""
            You are an AI assistant that helps with summarizing PDF documents.

            Here is the content of the PDF file '{pdf_file}':

            {text}

            Please summarize the content of this document in a few sentences.
            """

            # Send the prompt and get the response using utility function
            try:
                summary = await generate_text_ollama(model=model, prompt=prompt)
                logging.info(f"Summary of the PDF file '{pdf_file}':\n{summary}\n")
            except Exception as e:
                logging.error(
                    f"An error occurred while summarizing the PDF file '{pdf_file}': {str(e)}"
                )

    # Split and chunk
    text_chunks = []
    for page in all_pages:
        # Use utility function for splitting text
        chunks = split_text_into_chunks(page.page_content)
        text_chunks.extend(chunks)

    logging.info(f"Number of text chunks: {len(text_chunks)}")


    # === Create Metadata for Text Chunks ===
    # Example metadata management (customize as needed)
    def add_metadata(chunks, doc_title):
        metadata_chunks = []
        for chunk in chunks:
            metadata = {
                "title": doc_title,
                "author": "US Business Bureau",  # Update based on document data
                "date": str(datetime.date.today()),
            }
            # Ensure chunk is a string for Document creation
            metadata_chunks.append(Document(page_content=chunk, metadata=metadata))
        return metadata_chunks


    # add metadata to text chunks
    metadata_text_chunks = add_metadata(text_chunks, "BOI US FinCEN")


    ## === Add Embeddings to Vector Database Chromadb ===
    # Use utility function for creating vector db
    vector_db = create_vector_db(
        chunks=metadata_text_chunks,
        embedding_model_name=EMBEDDING_MODEL,
        collection_name="docs-local-rag",
        persist_directory=vector_db_path,
        use_fastembed=True # Use FastEmbedEmbeddings as per original file
    )

