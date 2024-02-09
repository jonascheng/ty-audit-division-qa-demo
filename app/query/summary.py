import logging
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain import hub

from util.openai import chatter


# Get logger
logger = logging.getLogger(__name__)


# function to summarization documents
def summarization(documents: []) -> str:
    """
    Summarization documents.
    """
    # create chatter
    llm = chatter()

    # replace "Helpful Answer:" with "請以中文提供有用的答案:"
    prompt_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify the main themes
    請以中文提供有用的答案:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # create summarization chain
    # llm_chain = load_summarize_chain(llm, chain_type="stuff")
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True)

    # execute summarization chain
    summary_dict = llm_chain.invoke(documents)
    summary_text = summary_dict["text"]
    logger.info(f'Summarized text: {summary_text}')

    return summary_text
