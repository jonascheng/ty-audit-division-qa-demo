import logging
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.retrievers import BaseRetriever

# Get logger
logger = logging.getLogger(__name__)


# function to create conversational retrieval qa
def create_conversational_retrieval_qa(
        llm,
        # memory,
        retriever: BaseRetriever,
        chain_type: str = 'stuff',
        return_source_documents: bool = False) -> ConversationalRetrievalChain:
    # create a chain to answer questions
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        # memory=memory,
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
            "chat_history": [],
        }
    )

    return search_results


# A class to do retrieval QA
class EmbeddingsRetrievalQA:
    def __init__(
            self,
            llm,
            retriever: BaseRetriever,
            chain_type: str = 'stuff',
            return_source_documents: bool = False):
        from util import stuff_prompt, map_reduce_prompt

        self.llm = llm
        self.retriever = retriever
        self.chain_type = chain_type
        self.return_source_documents = return_source_documents

        logger.info(
            f"EmbeddingsRetrievalQA: chain_type={chain_type}, return_source_documents={return_source_documents}")

        chain_type_kwargs = {
            'verbose': True
        }

        if chain_type == 'map_reduce':
            question_prompt = map_reduce_prompt.QUESTION_PROMPT_SELECTOR.get_prompt(llm)
            combine_prompt = map_reduce_prompt.COMBINE_PROMPT_SELECTOR.get_prompt(llm)
            chain_type_kwargs = {
                'question_prompt': question_prompt,
                'combine_prompt': combine_prompt,
                'verbose': True}
            logger.info(
                f"EmbeddingsRetrievalQA:\nquestion_prompt={question_prompt},\ncombine_prompt={combine_prompt}")
        elif chain_type == 'stuff':
            prompt = stuff_prompt.PROMPT_SELECTOR.get_prompt(llm)
            chain_type_kwargs = {
                'prompt': prompt,
                'verbose': True}
            logger.info(f"EmbeddingsRetrievalQA:\nprompt={prompt}")

        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type=chain_type,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=return_source_documents,
            verbose=True)

    # function to query by retrieval qa
    def query(self, query: str) -> list[dict]:
        if not query:
            return "Please provide a query."

        # get relevant documents
        search_results = self.qa({"query": query})

        return search_results
