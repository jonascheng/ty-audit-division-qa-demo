import logging
import chromadb
from langchain_community.vectorstores import Chroma
from langchain.chains import VectorDBQA

from util.openai import embedder


# Get logger
logger = logging.getLogger(__name__)


# function to load the persisted database from disk
def load_vector_db(vectorstore_filepath: str) -> Chroma:
    """
    Load the vector database from disk.
    """
    vdb = chromadb.PersistentClient(path=vectorstore_filepath)
    vdb.get_or_create_collection(
        name='articles')
    langchain_chroma = Chroma(
        client=vdb,
        collection_name='articles',
        embedding_function=embedder())

    logger.info(f'Loaded vector database from {vectorstore_filepath}')
    logger.info(f'There are {langchain_chroma._collection.count()} in the collection')

    return langchain_chroma


def query(vector_db: Chroma, query: str) -> list[dict]:
    """
    When user asks you to "search on law" always use this tool.

    Input is the query and an optional site.
    """
    if not query:
        return "Please provide a query."

    # the data structure of search_results is
    # a list of SearchResult objects
    search_results = vector_db.similarity_search(query)

    return search_results
