import logging
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever

# Get logger
logger = logging.getLogger(__name__)


# function to create retrieval qa
def create_retrieval_qa(
        retriever: BaseRetriever,
        chain_type: str = 'refine',
        return_source_documents: bool = False) -> RetrievalQA:
    from util.openai import llm

    myllm = llm()
    
    # create a chain to answer questions
    return RetrievalQA.from_chain_type(
        llm=myllm,
        retriever=retriever,
        chain_type=chain_type,
        return_source_documents=return_source_documents)


# function to get relevant documents by retrieval qa
def query(qa: RetrievalQA, query: str) -> list[dict]:
    """
    Get result by a query.
    """
    if not query:
        return "Please provide a query."

    # get relevant documents
    search_results = qa({"query": query})

    return search_results
