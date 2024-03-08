import os
import logging
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_extras.mention import mention
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

import yaml
from yaml.loader import SafeLoader

# Import proprietory module
import config.env
import config.logging
from dto import const
from query import web, embeddings, qa
from util.openai import chatter, memory

# Get logger
logger = logging.getLogger(__name__)


# Return indexer base on target name
def get_indexer(target_name: str):
    from query.embeddings import QueryEmbeddings

    if target_name == 'investigation':
        collection_name = os.environ.get(
            'EMBEDDINGS_INVESTIGATION_REPORTS_COLLECTION_NAME')
    else:
        collection_name = os.environ.get(
            'EMBEDDINGS_TAIWAN_LAW_COLLECTION_NAME')

    return QueryEmbeddings(
        vectorstore_filepath=os.environ.get('EMBEDDINGS_FILEPATH'),
        collection_name=collection_name)


# Function for querying Taiwan law database
def search_vector_store(
        prompt_input,
        indexer,
        top_k: int = 10,
        chain_type: str = 'stuff',) -> str:
    from query import qa
    from util.openai import chatter

    # create retrieval qa
    rqa = qa.EmbeddingsRetrievalQA(
        llm=chatter(),
        chain_type=chain_type,
        retriever=indexer.as_multiquery_retriever(
            top_k=top_k,
        ),
        return_source_documents=True)

    search_results = rqa.query({"query": prompt_input})

    return {
        "answer": search_results['result'],
        "source_documents": search_results['source_documents']
    }


def handle_selectbox_change():
    if st.session_state.target_name == const.APP_QUERY_TARGET_LAW:
        if "law_indexer" not in st.session_state.keys():
            st.session_state.law_indexer = get_indexer('law')
    elif st.session_state.target_name == const.APP_QUERY_TARGET_INVESTIGATION:
        if "investigation_indexer" not in st.session_state.keys():
            st.session_state.investigation_indexer = get_indexer('investigation')


def login():
    with st.sidebar:
        st.title(const.APP_TITLE)

        # a drop down for selecting the query target vector store
        target_name = st.selectbox(
            "選擇查詢目標",
            const.APP_QUERY_TARGETS,
            index=None,
            key='target_name',
            on_change=handle_selectbox_change)

        # an input for the user to enter a query
        query_input = st.text_input("輸入查詢", key='query_input')

        # a button to submit the query
        submit_button = st.button("送出查詢", key='submit_button')

    # Check if button is clicked
    if submit_button:
        if not target_name or not query_input:
            st.error("請選擇查詢目標並輸入查詢")
        else:
            with st.spinner("檢索中..."):
                if st.session_state.target_name == const.APP_QUERY_TARGET_LAW:
                    result_set = search_vector_store(
                        prompt_input=query_input,
                        indexer=st.session_state.law_indexer,
                        top_k=10,
                        chain_type='stuff')
                else:
                    result_set = search_vector_store(
                        prompt_input=query_input,
                        indexer=st.session_state.investigation_indexer,
                        top_k=5,
                        chain_type='map_reduce')

            # st.write(result_set)

            # Output the result in following format,
            # 檢索摘要： result_set['answer']
            # 資料來源：
            #     * result_set['source_documents'][0].metadata['source']
            #       result_set['source_documents'][0].page_content
            st.write(f"檢索摘要： {result_set['answer']}")
            # iterate through the source documents and display them
            st.write(f"資料來源：")
            for source_document in result_set['source_documents']:
                source = source_document.metadata['source']
                # if source is a link, display it as a link
                if source.startswith('http'):
                    st.write(f"    * {source}")
                else:
                    # source is a file path, only display the file name
                    file = os.path.basename(source)
                    st.write(f"    * {file}")
                content = source_document.page_content
                # remove all new lines from the content
                content = content.replace('\n', ' ')
                st.write(f"      {content} (省略部分內容...)")

st.set_page_config(page_title=const.APP_TITLE, page_icon='💬')

# authenticate user
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login()

if authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
else:
    login()
