import logging
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import MergerRetriever

from util.openai import embedder


# Get logger
logger = logging.getLogger(__name__)


# function to load the persisted database from disk
def load_vector_db(
        vectorstore_filepath: str,
        collection_name_prefix: str = 'law',
        collection_partition_size: int = 4) -> []:
    """
    Load the vector database from disk.
    """
    vdb = chromadb.PersistentClient(path=vectorstore_filepath)
    # list collection names
    collection_names = vdb.list_collections()
    logger.info(f'Collection names: {collection_names}')

    langchain_chromas = []
    for i in range(collection_partition_size):
        collection_name = f'{collection_name_prefix}_{i}'
        # check if collection exists
        # vdb.get_collection(name=collection_name)
        langchain_chroma = Chroma(
            client=vdb,
            collection_name=collection_name,
            embedding_function=embedder())
        logger.info(f'There are {langchain_chroma._collection.count()} in the collection {collection_name}')
        langchain_chromas.append(langchain_chroma)

    logger.info(f'Loaded vector database from {vectorstore_filepath}')

    return langchain_chromas


# function to query similar documents
def similarity_search(langchain_chromas: [], query: str) -> []:
    """
    Query similar documents by Chromas.
    """
    if not query:
        return "Please provide a query."

    # the data structure of search_results is
    # a list of SearchResult objects
    search_results = []
    for langchain_chroma in langchain_chromas:
        search_results.append(langchain_chroma.similarity_search(query))

    return search_results


# function to create merger retriever
def create_merger_retriever(langchain_chromas: []) -> MergerRetriever:
    """
    Create merger retriever.
    """
    return MergerRetriever(
        retrievers=[langchain_chroma.as_retriever() for langchain_chroma in langchain_chromas])


def get_relevant_documents(retriever: BaseRetriever, query: str) -> list[dict]:
    """
    Get relevant documents by retriever.
    """
    if not query:
        return "Please provide a query."

    # the data structure of search_results is
    # a list of Document objects
    search_results = retriever.get_relevant_documents(query)

    logger.info(f'Found {len(search_results)} relevant documents')

    return search_results
