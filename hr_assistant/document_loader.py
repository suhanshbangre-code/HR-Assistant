from langchain_community.document_loaders import TextLoader
from hr_assistant import config
from hr_assistant.logger import get_logger
logger = get_logger(__name__)

def load_document(file_path: str = config.DATA_FILE_PATH):
    """Load a .txt file and return it as a list of Langchain documents"""
    logger.info("Loading documents from ", file_path)

    text_loader = TextLoader(file_path, encoding="utf-8")
    documents = text_loader.load()

    logger.info(f"Loaded {len(documents)} document(s)")
    return documents