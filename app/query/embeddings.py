import logging
import chromadb
from typing import Union, List
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import MergerRetriever

from util.openai import embedder


# Get logger
logger = logging.getLogger(__name__)


# function to load vector store with single collection from disk
def load_vector_db(
        vectorstore_filepath: str,
        collection_name: str) -> Chroma:
    """
    Load the vector database from disk.
    """
    vdb = chromadb.PersistentClient(path=vectorstore_filepath)
    # list collection names
    collection_names = vdb.list_collections()
    logger.info(f'Collection names: {collection_names}')

    langchain_chroma = Chroma(
        client=vdb,
        collection_name=collection_name,
        embedding_function=embedder())
    logger.info(
        f'There are {langchain_chroma._collection.count()} in the collection {collection_name}')

    logger.info(f'Loaded vector database from {vectorstore_filepath}')

    return langchain_chroma


# function to create merger retriever
def create_merger_retriever(
        langchain_chromas: [],) -> MergerRetriever:
    """
    Create merger retriever.
    """
    return MergerRetriever(
        retrievers=[
            langchain_chroma.as_retriever(
                search_kwargs={'k': 5})
            for langchain_chroma in langchain_chromas])


# A class to query embeddings
class QueryEmbeddings:
    def __init__(
            self,
            vectorstore_filepath: str,
            collection_name: str):
        import chromadb
        from langchain_community.vectorstores import Chroma

        # load vector store with single collection from disk
        self.vdb = chromadb.PersistentClient(
            path=vectorstore_filepath)
        # list collection names
        collection_names = self.vdb.list_collections()
        logger.debug(f'Collection names: {collection_names}')

        self.store = Chroma(
            client=self.vdb,
            collection_name=collection_name,
            embedding_function=embedder())
        logger.info(
            f'There are {self.store._collection.count()} in the collection {collection_name}')

    def similarity_search(self, query: str) -> Union[str, List[dict]]:
        # query similar documents
        if not query:
            return "Please provide a query."

        # the data structure of search_results is
        # a list of SearchResult objects along with scores
        search_results = self.store.similarity_search_with_score(query)
        # search_results = self.store.similarity_search(query)
        logger.info(
            f'Found {len(search_results)} similar documents with query: {query}')

        return search_results

    # function to return retriever
    def as_retriever(self) -> BaseRetriever:
        return self.store.as_retriever()
