import os
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI
from langchain_openai.llms import OpenAI, AzureOpenAI


# embedder function
def embedder():
    # if not azure type
    if os.environ.get('OPENAI_API_TYPE') != 'azure':
        return OpenAIEmbeddings(
            retry_min_seconds=60,
            retry_max_seconds=600,
            max_retries=10)
    # if azure type
    if os.environ.get('OPENAI_API_TYPE') == 'azure':
        return AzureOpenAIEmbeddings(
            azure_deployment='text-search-davinci-doc-001',
            model='text-search-davinci-doc-001',
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            openai_api_type=os.environ.get('OPENAI_API_TYPE'),
            api_key=os.environ.get('OPENAI_API_KEY'),
            api_version=os.environ.get('OPENAI_API_VERSION'),
            chunk_size=1)


# llm function
def llm():
    # if not azure type
    if os.environ.get('OPENAI_API_TYPE') != 'azure':
        return OpenAI(
            retry_min_seconds=60,
            retry_max_seconds=600,
            max_retries=10,
            verbose=True)
    # if azure type
    if os.environ.get('OPENAI_API_TYPE') == 'azure':
        deployment_name = "text-davinci-003"
        return AzureOpenAI(
            deployment_name=deployment_name,
            model=deployment_name,
            temperature=0,
            frequency_penalty=0.2,
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            openai_api_type=os.environ.get('OPENAI_API_TYPE'),
            api_version=os.environ.get('OPENAI_API_VERSION'),
            api_key=os.environ.get('OPENAI_API_KEY'),
            verbose=True)


# chat function
def chatter():
    # if not azure type
    if os.environ.get('OPENAI_API_TYPE') != 'azure':
        return ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo-1106",
            retry_max_seconds=600,
            max_retries=10,
            verbose=True)
    # if azure type
    if os.environ.get('OPENAI_API_TYPE') == 'azure':
        deployment_name = "gpt-35-turbo"
        return AzureOpenAI(
            deployment_name=deployment_name,
            model=deployment_name,
            temperature=0,
            frequency_penalty=0.2,
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            openai_api_type=os.environ.get('OPENAI_API_TYPE'),
            api_version=os.environ.get('OPENAI_API_VERSION'),
            api_key=os.environ.get('OPENAI_API_KEY'),
            verbose=True)
