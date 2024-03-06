import logging
import chromadb
from typing import Union, List
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import MergerRetriever

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from util.openai import embedder, chatter

# Get logger
logger = logging.getLogger(__name__)


# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


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

    # function to query similar documents
    def similarity_search(
            self,
            query: str,
            score_threshold: float = 0.2) -> Union[str, List[dict]]:
        if not query:
            return "Please provide a query."

        # the data structure of search_results is
        # a list of SearchResult objects along with scores
        # search_results = self.store.similarity_search_with_score(query)
        search_results = self.store.similarity_search_with_relevance_scores(
            query,
            score_threshold=score_threshold,)
        # search_results = self.store.similarity_search(query)
        logger.info(
            f'Found {len(search_results)} similar documents with query: {query}')

        return search_results

    # function to return retriever
    def as_retriever(
            self,
            score_threshold: float = 0.2,
            top_k: int = 10) -> BaseRetriever:
        return self.store.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={
                "score_threshold": score_threshold,
                "k": top_k,},
        )

    # function to return multiquery retriever
    def as_multiquery_retriever(
            self,
            score_threshold: float = 0.2,
            top_k: int = 10) -> BaseRetriever:
        from langchain.chains.llm import LLMChain
        from langchain.retrievers.multi_query import MultiQueryRetriever

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from a vector
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search.
            Provide these alternative questions in Traditional Chinese and separated by newlines.
            Original question: {question}""",
        )
        output_parser = LineListOutputParser()

        llm = chatter()

        # create chain
        llm_chain = LLMChain(
            llm=llm,
            prompt=QUERY_PROMPT,
            output_parser=output_parser,
            verbose=True,)

        return MultiQueryRetriever(
            llm_chain=llm_chain,
            retriever=self.store.as_retriever(
                search_type='similarity_score_threshold',
                search_kwargs={
                    "score_threshold": score_threshold,
                    "k": top_k, }),
            # "lines" is the key (attribute name) of the parsed output
            parser_key="lines",
            verbose=True,)
