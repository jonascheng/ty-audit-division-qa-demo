import os
import logging
import streamlit as st
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Import proprietory module
import config.env
import config.logging
from dto import const
from query import web
from util.openai import llm

# Get logger
logger = logging.getLogger(__name__)


# Function for querying Taiwan law database by use cases
def search_taiwan_law_db_by_use_cases(prompt_input) -> str:
    result_set = web.google_search(
        prompt_input,
        site_name='台灣全國法規資料庫 智慧查找案例',
        site_link='https://law.moj.gov.tw/SmartSearch/Theme.aspx')
    return result_set


# Function for querying Taiwan law database
def search_taiwan_law_db(prompt_input) -> str:
    from query import embeddings, qa
    from util.openai import chatter

    # load vector database from disk for taiwan law
    law_vdb = embeddings.load_vector_db(
        vectorstore_filepath=os.environ.get('EMBEDDINGS_FILEPATH'),
        collection_name=os.environ.get('EMBEDDINGS_COLLECTION_NAME'))
    # merge vdbs into langchain_chromas
    langchain_chromas = [law_vdb]
    # create merger retriever
    retriever = embeddings.create_merger_retriever(langchain_chromas)
    # create retrieval qa
    rqa = qa.create_retrieval_qa(
        llm=chatter(),
        retriever=retriever,
        return_source_documents=False)
    # get relevant documents
    search_results = qa.query(rqa, prompt_input)

    return search_results['result']


st.set_page_config(page_title=const.APP_TITLE, page_icon='💬')

with st.sidebar:
    st.title(const.APP_TITLE)
    st.write('Hello world!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        SystemMessage(content="你是台灣審計部AI小助手，請以繁體中文回答審計人員提問"),
        AIMessage(content="你好，我可以協助你關於台灣法規及相關案例檢索，有什麼可以協助你的嗎?")
    ]

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)
    elif isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)

# User-provided prompt
if prompt := st.chat_input(placeholder="請輸入你的問題"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

# Generate a new response if last message is not from assistant
if not isinstance(st.session_state.messages[-1], AIMessage):
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = search_taiwan_law_db(prompt)
            st.write(response)
    message = AIMessage(content=response)
    st.session_state.messages.append(message)
