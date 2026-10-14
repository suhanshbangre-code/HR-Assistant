"""Store chunk embeddings in FAISS so that we can search them later"""

import os

# from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model
from hr_assistant.logger import get_logger
logger = get_logger(__name__)


# building vector store
def build_vector_store(chunks):
    """Embed every chunk and upload it into Qdrant Cloud collection"""
    logger.info(f"Embedding {len(chunks)} chunk(s) and uploading it to Qdrant Collection {config.QDRANT_COLLECTION_NAME}...")
    embeddings_model = get_embeddings_model()
    vector_store = QdrantVectorStore.from_documents(
        chunks, 
        embedding=embeddings_model,
        url = config.QDRANT_URL,
        api_key = config.QDRANT_API_KEY,
        collection_name = config.QDRANT_COLLECTION_NAME
    )

    logger.info(f"Uploaded to Qdrant Collection {config.QDRANT_COLLECTION_NAME}")
    return vector_store


# load vector store
def load_vector_store():
    """Connect to Qdrant Cloud collection that was already built before."""
    logger.info("Connecting to Qdrant Cloud")

    embeddings_model = get_embeddings_model()
    return QdrantVectorStore.from_documents(
            embedding=embeddings_model,
            url = config.QDRANT_URL,
            api_key = config.QDRANT_API_KEY,
            collection_name = config.QDRANT_COLLECTION_NAME
        )


# check if vector store exists
def vector_store_exists() -> bool:
    """Check if a Qdrant store already exists"""
    client = QdrantClient(
       url=config.QDRANT_URL,
       api_key=config.QDRANT_API_KEY 
    )
    return client.collection_exists(config.QDRANT_COLLECTION_NAME)


def get_retriever(vector_store, k: int = config.TOP_K_RESULT):
    """Turn a vector store into a retriever that returns the top-k matching chunks."""
    logger.info(f"Creating a retriever with top_k = {k}")
    return vector_store.as_retriever(search_kwargs = {"k":k})