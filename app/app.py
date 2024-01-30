import os
import logging
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.retrievers import MergerRetriever

# Import proprietory module
import config.env
import config.logging
from dto import const
from query import web
from tools import embeddings
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
    myllm = llm()
    law_vdb = embeddings.load_vector_db(
        vectorstore_filepath=os.environ.get('EMBEDDINGS_LAW_FILEPATH'))
    order_vdb = embeddings.load_vector_db(
        vectorstore_filepath=os.environ.get('EMBEDDINGS_ORDER_FILEPATH'))
    # The Lord of the Retrievers will hold the output of both retrievers and can be used as any other
    # retriever on different types of chains.
    lotr = MergerRetriever(
        retrievers=[law_vdb.as_retriever(), order_vdb.as_retriever()])
    qa = RetrievalQA.from_chain_type(
        myllm,
        retriever=lotr,
        chain_type='stuff',
        return_source_documents=True,
        verbose=True,
    )
    result_set = qa({'query': prompt_input})
    return result_set


st.set_page_config(page_title=const.APP_TITLE, page_icon='💬')

with st.sidebar:
    st.title(const.APP_TITLE)
    st.write('Hello world!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "你好，我是AI小助手，我可以協助你關於台灣法規及相關案例檢索，有什麼我可以協助你的嗎?"
        }
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input(placeholder="請輸入你的問題"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # response = search_taiwan_law_db_by_use_cases(prompt)
            response = search_taiwan_law_db(prompt)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
