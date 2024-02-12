import logging
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.retrievers import BaseRetriever

# Get logger
logger = logging.getLogger(__name__)


# function to create retrieval qa
def create_retrieval_qa(
        llm,
        retriever: BaseRetriever,
        chain_type: str = 'stuff',
        return_source_documents: bool = False) -> RetrievalQA:
    # create a chain to answer questions
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type=chain_type,
        return_source_documents=return_source_documents,
        verbose=True)


# function to get relevant documents by retrieval qa
def query_by_retrieval_qa(qa: RetrievalQA, query: str) -> list[dict]:
    """
    Get result by a query.
    """
    if not query:
        return "Please provide a query."

    # get relevant documents
    search_results = qa({"query": query})

    return search_results


# function to create conversational retrieval qa
def create_conversational_retrieval_qa(
        llm,
        memory,
        retriever: BaseRetriever,
        chain_type: str = 'stuff',
        return_source_documents: bool = False) -> ConversationalRetrievalChain:
    # create a chain to answer questions
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        return_source_documents=return_source_documents,
        verbose=True)


# function to get relevant documents by conversational retrieval qa
def query_by_conversational_retrieval_qa(
        qa: ConversationalRetrievalChain,
        question: str,) -> list[dict]:
    """
    Get result by a query.
    """
    if not question:
        return "Please provide a query."

    # get relevant documents
    search_results = qa(
            {
                "question": question,
            }
        )

    return search_results
