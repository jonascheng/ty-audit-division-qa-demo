import os
import logging
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_openai.llms import OpenAI, AzureOpenAI

# Get logger
logger = logging.getLogger(__name__)


# embedding function
def embedder():
    from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

    # if not azure type
    if os.environ.get('OPENAI_API_TYPE') != 'azure':
        return OpenAIEmbeddings(
            model=os.environ.get('OPENAI_EMBEDDING_MODEL'),
            retry_min_seconds=60,
            retry_max_seconds=600,
            max_retries=10)
    # if azure type
    if os.environ.get('OPENAI_API_TYPE') == 'azure':
        return AzureOpenAIEmbeddings(
            azure_deployment=os.environ.get('AZURE_EMBEDDING_DEPLOYMENT'),
            model=os.environ.get('OPENAI_EMBEDDING_MODEL'),
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            openai_api_type=os.environ.get('OPENAI_API_TYPE'),
            api_key=os.environ.get('OPENAI_API_KEY'),
            api_version=os.environ.get('OPENAI_API_VERSION'),
            chunk_size=1)


# split documents into chunks function
def text_splitter(
        documents,
        chunk_size=900,
        chunk_overlap=0,
        separators=[r"\s+", "。", "　", "＞"],):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        # separators=["\n\n", "\r\n", "\r", "\n", " ", "。", "　"],
        separators=separators,
        is_separator_regex=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    return chunks


# token and cost estimation in USD function
def calculate_embedding_cost(documents) -> (int, float):
    import tiktoken

    model_name = os.environ.get('OPENAI_EMBEDDING_MODEL')

    # a map to encoding name for different models
    model_encoding = {
        'text-embedding-3-small': 'cl100k_base',
        'text-embedding-3-large': 'cl100k_base',
    }
    # set default encoding name to None
    encoding_name = model_encoding.get(model_name, None)
    # get encoding
    if encoding_name is None:
        enc = tiktoken.encoding_for_model(model_name=model_name)
    else:
        enc = tiktoken.get_encoding(encoding_name)

    # a map to price for different models, in dollars per 1000 tokens
    model_price = {
        'text-embedding-3-small': 0.00002,
        'text-embedding-3-large': 0.00013,
    }
    # set default price to 0.0001
    price = model_price.get(model_name, 0.0001)
    logger.info(f'Using model {model_name}, encoding {enc.name}, price ${price:.5f} per 1000 tokens')

    total_tokens = sum([len(enc.encode(page.page_content)) for page in documents])

    return total_tokens, total_tokens / 1000 * price


# llm function
def llm():
    temperature = 0
    frequency_penalty = 0.2
    # if not azure type
    if os.environ.get('OPENAI_API_TYPE') != 'azure':
        return OpenAI(
            model=os.environ.get('OPENAI_LLM_MODEL'),
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            max_retries=10,
            verbose=True)
    # if azure type
    if os.environ.get('OPENAI_API_TYPE') == 'azure':
        return AzureOpenAI(
            deployment_name=os.environ.get('AZURE_LLM_DEPLOYMENT'),
            model=os.environ.get('OPENAI_LLM_MODEL'),
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            openai_api_type=os.environ.get('OPENAI_API_TYPE'),
            api_version=os.environ.get('OPENAI_API_VERSION'),
            api_key=os.environ.get('OPENAI_API_KEY'),
            verbose=True)


# chat function
def chatter():
    temperature = 0
    max_tokens = 512
    # if not azure type
    if os.environ.get('OPENAI_API_TYPE') != 'azure':
        return ChatOpenAI(
            model=os.environ.get('OPENAI_CHAT_MODEL'),
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=10,
            verbose=True)
    # if azure type
    if os.environ.get('OPENAI_API_TYPE') == 'azure':
        return AzureChatOpenAI(
            deployment_name=os.environ.get('AZURE_CHAT_DEPLOYMENT'),
            model=os.environ.get('OPENAI_CHAT_MODEL'),
            temperature=temperature,
            max_tokens=max_tokens,
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            openai_api_type=os.environ.get('OPENAI_API_TYPE'),
            api_version=os.environ.get('OPENAI_API_VERSION'),
            api_key=os.environ.get('OPENAI_API_KEY'),
            verbose=True)
