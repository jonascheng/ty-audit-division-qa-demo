import os
import logging
import streamlit as st
from streamlit_extras.mention import mention
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Import proprietory module
import config.env
import config.logging
from dto import const
from query import web, embeddings, qa
from util.openai import chatter, memory

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
def search_taiwan_law_db(
        prompt_input,
        vector_db,
        chat_history) -> str:
    # merge vdbs into langchain_chromas
    langchain_chromas = [vector_db]
    # create merger retriever
    retriever = embeddings.create_merger_retriever(langchain_chromas)
    # create retrieval qa
    rqa = qa.create_conversational_retrieval_qa(
        llm=chatter(),
        # memory=chat_history,
        retriever=retriever,
        return_source_documents=True)
    # get relevant documents
    search_results = qa.query_by_conversational_retrieval_qa(
        rqa,
        prompt_input,)
    # sources = ", ".join(search_results['source_documents'])
    # return f"{search_results['answer']}\nsources: {sources}"
    return search_results


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
    st.session_state.chat_history = memory(
        llm=chatter(),
        memory_key='chat_history',
        return_messages=True)
    # load vector database from disk for taiwan law
    st.session_state.vdb = embeddings.load_vector_db(
        vectorstore_filepath=os.environ.get('EMBEDDINGS_FILEPATH'),
        collection_name=os.environ.get('EMBEDDINGS_COLLECTION_NAME'))

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
            if message.additional_kwargs:
                for doc in message.additional_kwargs["source_documents"]:
                    mention(
                        label=f'{doc.metadata["law_name"]} {doc.metadata["law_article_chapter"]} {doc.metadata["law_article_no"]}',
                        icon="📌",
                        url=doc.metadata["source"],
                    )
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
            response = search_taiwan_law_db(
                prompt,
                st.session_state.vdb,
                st.session_state.chat_history)
            st.write(response["answer"])
            for doc in response["source_documents"]:
                mention(
                    label=f'{doc.metadata["law_name"]} {doc.metadata["law_article_chapter"]} {doc.metadata["law_article_no"]}',
                    icon="📌",
                    url=doc.metadata["source"],
                )
    message = AIMessage(
        content=response["answer"],
        additional_kwargs={"source_documents": response["source_documents"]})
    st.session_state.messages.append(message)
