"""Store chunk embeddings in FAISS so that we can search them later"""

import os
from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model

# building vector store
def build_vector_store(chunks):
    """Embed every chunk and build a searchable FAISS index in memeory"""
    embeddings_model = get_embeddings_model()
    return FAISS.from_documents(chunks, embeddings_model)

# save vector store
def save_vector_store(vector_store, path: str = config.VECTOR_STORE_PATH) -> None:
    """Save the FAISS index to disk so that we don't have to rebuild it every time."""
    vector_store.save_local(path)

# load vector store
def load_vector_store(path: str = config.VECTOR_STORE_PATH):
    """Load a previously saved FAISS index from disk"""
    embeddings_model = get_embeddings_model()
    return FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization = True)

# check if vector store exists
def vector_store_exists(path: str = config.VECTOR_STORE_PATH) -> bool:
    """Check if a saved FAISS index already exists on disk"""
    return os.path.exists(os.path.join(path, "index.faiss"))

def get_retriever(vector_store, k: int = config.TOP_K_RESULT):
    """Turn a vector store into a retriever that returns the top-k matching chunks."""
    return vector_store.as_retriever(search_kwargs = {"k":k})