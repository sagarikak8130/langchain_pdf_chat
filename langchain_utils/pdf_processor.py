from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

def process_pdf(filepath):
    logger.info(f"Starting PDF processing for file: {filepath}")
    
    # Load PDF
    logger.info("Loading PDF document")
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    logger.info(f"PDF loaded successfully with {len(pages)} pages")

    # Split text
    logger.info("Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(pages)
    logger.info(f"Text split into {len(chunks)} chunks")

    # Create vector store
    logger.info("Creating vector store with FAISS")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("Vector store created successfully")

    return vectorstore