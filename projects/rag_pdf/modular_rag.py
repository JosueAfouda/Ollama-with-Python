# main.py

import os
import logging
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

from src.utils.rag_utils import ingest_pdf, split_documents, create_vector_db
from src.utils.ollama_utils import pull_ollama_model

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "./data/BOI.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created successfully.")
    return chain


def main():
    # Load and process the PDF document
    data = ingest_pdf(DOC_PATH) # Use utility function
    if data is None:
        return

    # Split the documents into chunks
    chunks = split_documents(data) # Use utility function

    # Create the vector database
    # Use utility function, passing embedding model name and collection name
    vector_db = create_vector_db(chunks, embedding_model_name=EMBEDDING_MODEL, collection_name=VECTOR_STORE_NAME)

    # Initialize the language model
    llm = ChatOllama(model=MODEL_NAME)

    # Create the retriever
    retriever = create_retriever(vector_db, llm)

    # Create the chain with preserved syntax
    chain = create_chain(retriever, llm)

    # Example query
    question = "How to report BOI?"

    # Get the response
    res = chain.invoke(input=question)
    print("Response:")
    print(res)


if __name__ == "__main__":
    main()
